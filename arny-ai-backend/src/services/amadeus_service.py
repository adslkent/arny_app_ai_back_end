"""
Enhanced Amadeus Service with Comprehensive Tenacity Retry Strategies

This module provides enhanced service for interacting with Amadeus APIs for flights and hotels
with comprehensive retry strategies using Tenacity library following 5 key conditions:

1. Check success flag (response.ok or equivalent) to retry on any non-2xx/3xx response
2. Inspect payload for "error"/"warning" fields, retrying when they're present  
3. Match on exception messages via retry_if_exception_message (timeout, failed, unavailable)
4. Combine predicates with retry_any to catch both exceptions and bad results
5. Validate against schema/model (Pydantic) and retry when validation fails
"""

from typing import List, Dict, Any, Optional
from amadeus import Client, ResponseError
import logging
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception_message,
    retry_any,
    retry_if_result,
    before_sleep_log,
    retry_if_exception
)
from pydantic import BaseModel, ValidationError

from ..utils.config import config

# Set up logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class AmadeusFlightOffer(BaseModel):
    """Pydantic model for flight offer validation"""
    id: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    price: Optional[Dict[str, Any]] = None
    itineraries: Optional[List[Dict[str, Any]]] = None
    validatingAirlineCodes: Optional[List[str]] = None

class AmadeusHotelOffer(BaseModel):
    """Pydantic model for hotel offer validation"""
    type: Optional[str] = None
    hotel: Optional[Dict[str, Any]] = None
    available: Optional[bool] = None
    offers: Optional[List[Dict[str, Any]]] = None

class AmadeusSearchResponse(BaseModel):
    """Pydantic model for search response validation"""
    success: bool
    results: Optional[List[Dict[str, Any]]] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

# ==================== CUSTOM RETRY CONDITIONS ====================

def retry_on_amadeus_error_result(result):
    """
    Condition 2: Inspect the payload for "error"/"warning" fields, retrying when they're present
    """
    if isinstance(result, dict):
        return (
            result.get("success") is False or
            result.get("error") is not None or
            result.get("error_code") is not None or
            "warning" in result or
            "error" in str(result).lower()
        )
    return False

def retry_on_amadeus_response_validation_failure(result):
    """
    Condition 5: Validate against schema/model (Pydantic) and retry when validation fails
    """
    if isinstance(result, dict) and result.get("success"):
        try:
            # Validate the response structure
            AmadeusSearchResponse(**result)
            
            # If it has results, validate individual items
            if result.get("results"):
                for item in result["results"][:1]:  # Validate first item as sample
                    if "itineraries" in item:  # Flight offer
                        AmadeusFlightOffer(**item)
                    elif "hotel" in item:  # Hotel offer
                        AmadeusHotelOffer(**item)
            return False  # Validation passed
        except ValidationError as e:
            logger.warning(f"Amadeus response validation failed: {e}")
            return True  # Validation failed, retry
        except Exception as e:
            logger.warning(f"Unexpected validation error: {e}")
            return True
    return False

def retry_on_amadeus_http_status(result):
    """
    Condition 1: Check success flag equivalent to retry on non-2xx/3xx responses
    """
    if isinstance(result, dict):
        error_code = result.get("error_code")
        if error_code:
            try:
                status_code = int(error_code)
                # Retry on 4xx and 5xx status codes, but not on 2xx/3xx
                return status_code >= 400
            except (ValueError, TypeError):
                # If error_code is not numeric, check if it indicates HTTP error
                return any(code in str(error_code).lower() for code in ['400', '401', '403', '404', '429', '500', '502', '503', '504'])
    return False

def retry_on_amadeus_exception_type(exception):
    """
    Custom exception checker for Amadeus-specific exceptions
    """
    return isinstance(exception, (ResponseError, requests.exceptions.RequestException, ConnectionError, TimeoutError))

# ==================== COMBINED RETRY STRATEGIES ====================

# Primary retry strategy for critical Amadeus operations (flights/hotels)
amadeus_critical_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|429|502|503|504|rate.?limit).*"),
        # Condition 4: Exception types (combining ResponseError and connection issues)
        retry_if_exception_type((ResponseError, requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        # Custom exception checker
        retry_if_exception(retry_on_amadeus_exception_type),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_amadeus_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_amadeus_http_status),
        # Condition 5: Validation failure
        retry_if_result(retry_on_amadeus_response_validation_failure)
    ),
    stop=stop_after_attempt(4),  # More attempts for critical operations
    wait=wait_exponential(multiplier=1.5, min=1, max=15),  # Longer max wait
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Secondary retry strategy for less critical operations (airport search, check-in links)
amadeus_secondary_retry = retry(
    retry=retry_any(
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|network|connection|429|502|503|504).*"),
        retry_if_exception_type((ResponseError, requests.exceptions.RequestException, ConnectionError, TimeoutError)),
        retry_if_exception(retry_on_amadeus_exception_type),
        retry_if_result(retry_on_amadeus_error_result),
        retry_if_result(retry_on_amadeus_http_status)
    ),
    stop=stop_after_attempt(3),  # Fewer attempts for secondary operations
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

class AmadeusService:
    """Service for interacting with Amadeus APIs with comprehensive retry strategies"""
    
    def __init__(self):
        try:
            self.client = Client(
                client_id=config.AMADEUS_API_KEY,
                client_secret=config.AMADEUS_API_SECRET,
                hostname=config.AMADEUS_BASE_URL
            )
            self.logger = logging.getLogger(__name__)
            logger.info("AmadeusService initialized with enhanced retry strategies")
        except Exception as e:
            logger.error(f"Failed to initialize AmadeusService: {e}")
            raise
    
    # ==================== FLIGHT OPERATIONS WITH RETRY STRATEGIES ====================
    
    @amadeus_critical_retry
    async def search_flights(self, origin: str, destination: str, departure_date: str, 
                           return_date: Optional[str] = None, adults: int = 1, 
                           cabin_class: str = "ECONOMY", max_results: int = 50) -> Dict[str, Any]:
        """
        Search for flights using Amadeus Flight Offers Search API with comprehensive retry strategies
        
        Enhanced with 5-condition retry strategy:
        1. HTTP status checking
        2. Error/warning field inspection  
        3. Exception message matching
        4. Combined exception and result predicates
        5. Pydantic validation with retry on failure
        """
        try:
            logger.info(f"Searching flights: {origin} -> {destination} on {departure_date} (with retry strategies)")
            
            # Prepare search parameters
            search_params = {
                'originLocationCode': origin,
                'destinationLocationCode': destination,
                'departureDate': departure_date,
                'adults': adults,
                'max': max_results,
                'travelClass': cabin_class
            }
            
            # Add return date if provided (for round-trip)
            if return_date:
                search_params['returnDate'] = return_date
            
            # Make API call with automatic retry on failures
            response = self.client.shopping.flight_offers_search.get(**search_params)
            
            # Process and format results
            flight_offers = []
            if hasattr(response, 'data') and response.data:
                for offer in response.data:
                    formatted_offer = self._format_flight_offer(offer)
                    flight_offers.append(formatted_offer)
            
            # Create response that will be validated by retry strategy
            result = {
                "success": True,
                "results": flight_offers,
                "meta": {
                    "count": len(flight_offers),
                    "search_params": search_params
                }
            }
            
            # Validate response structure (triggers retry if validation fails)
            try:
                AmadeusSearchResponse(**result)
                logger.info(f"Flight search successful: {len(flight_offers)} results found")
            except ValidationError as ve:
                logger.warning(f"Flight search response validation failed: {ve}")
                # Return error result to trigger retry
                return {
                    "success": False,
                    "error": f"Response validation failed: {str(ve)}",
                    "results": []
                }
            
            return result
            
        except ResponseError as e:
            error_msg = f"Amadeus flight search API error: {str(e)}"
            logger.error(error_msg)
            
            # Properly extract status code from Response object
            status_code = 'unknown'
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = str(e.response.status_code)
            
            result = {
                "success": False,
                "error": error_msg,
                "error_code": status_code,
                "results": []
            }
            
            # Let retry strategy handle this error
            return result
            
        except Exception as e:
            error_msg = f"Unexpected flight search error: {str(e)}"
            logger.error(error_msg)
            
            result = {
                "success": False,
                "error": error_msg,
                "results": []
            }
            
            return result

    @amadeus_critical_retry
    async def get_flight_price(self, flight_offer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get accurate pricing for a specific flight offer with retry strategies
        """
        try:
            logger.info("Getting flight pricing with retry strategies")
            
            # The flight_offer should be the complete offer object from search
            response = self.client.shopping.flight_offers.pricing.post(flight_offer)
            
            if hasattr(response, 'data') and response.data:
                # Format the pricing response
                priced_offer = response.data.get('flightOffers', [{}])[0]
                
                result = {
                    "success": True,
                    "priced_offer": self._format_flight_offer(priced_offer),
                    "booking_requirements": response.data.get('bookingRequirements', {}),
                    "pricing_valid_until": response.data.get('expirationDateTime')
                }
                
                # Validate response
                try:
                    if result["priced_offer"]:
                        AmadeusFlightOffer(**result["priced_offer"])
                    logger.info("Flight pricing successful")
                except ValidationError as ve:
                    logger.warning(f"Flight pricing validation failed: {ve}")
                    return {
                        "success": False,
                        "error": f"Pricing validation failed: {str(ve)}"
                    }
                
                return result
            else:
                return {
                    "success": False,
                    "error": "No pricing data returned"
                }
                
        except ResponseError as e:
            error_msg = f"Amadeus flight pricing API error: {str(e)}"
            logger.error(error_msg)
            
            # Properly extract status code from Response object
            status_code = 'unknown'
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = str(e.response.status_code)
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": status_code
            }
            
        except Exception as e:
            logger.error(f"Unexpected flight pricing error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    # ==================== HOTEL OPERATIONS WITH RETRY STRATEGIES ====================

    @amadeus_critical_retry
    async def search_hotels(self, city_code: str, check_in_date: str, check_out_date: str,
                        adults: int = 1, rooms: int = 1, max_results: int = 50) -> Dict[str, Any]:
        """
        Search for hotels using multiple Amadeus Hotel API approaches with comprehensive error handling
        """
        logger.info(f"🏨 Starting hotel search for {city_code} with multiple approaches")
        
        # Try multiple approaches to find hotels
        approaches = [
            self._search_hotels_approach_1,  # Direct hotel offers by city
            self._search_hotels_approach_2,  # Two-step: hotel list + offers
            self._search_hotels_approach_3,  # Geocode-based search
        ]
        
        for i, approach in enumerate(approaches, 1):
            try:
                logger.info(f"🔄 Trying hotel search approach {i} for {city_code}")
                result = await approach(city_code, check_in_date, check_out_date, adults, rooms, max_results)
                
                if result.get("success") and result.get("results"):
                    # SUCCESS LOGGING
                    approach_name = result.get("meta", {}).get("approach", f"approach_{i}")
                    hotel_count = len(result["results"])
                    logger.info(f"✅ Hotel search approach {i} ({approach_name}) SUCCESS: {hotel_count} hotels found for {city_code}")
                    print(f"✅ Hotel search approach {i} ({approach_name}) SUCCESS: {hotel_count} hotels found for {city_code}")
                    return result
                else:
                    # FAILURE LOGGING
                    error_msg = result.get('error', 'No error message')
                    logger.warning(f"⚠️ Hotel search approach {i} FAILED for {city_code}: {error_msg}")
                    print(f"⚠️ Hotel search approach {i} FAILED for {city_code}: {error_msg}")
                    
            except Exception as e:
                # EXCEPTION LOGGING
                logger.error(f"❌ Hotel search approach {i} EXCEPTION for {city_code}: {str(e)}")
                print(f"❌ Hotel search approach {i} EXCEPTION for {city_code}: {str(e)}")
                continue
        
        # If all approaches fail, return a structured error
        logger.error(f"❌ ALL hotel search approaches FAILED for {city_code}")
        print(f"❌ ALL hotel search approaches FAILED for {city_code}")
        
        return {
            "success": False,
            "error": f"All hotel search approaches failed for city {city_code}. This might be an API issue or invalid city code.",
            "results": [],
            "suggestions": [
                "Try searching with a different city name",
                "Check if the dates are valid and in the future",
                "Try again in a few minutes as this might be a temporary API issue"
            ]
        }

    async def _search_hotels_approach_1(self, city_code: str, check_in_date: str, check_out_date: str,
                                    adults: int, rooms: int, max_results: int) -> Dict[str, Any]:
        """
        Approach 1: Try direct city-based hotel search using different endpoint patterns
        """
        try:
            # Try different potential endpoint structures
            endpoints_to_try = [
                # Try the standard hotel offers endpoint with city code
                lambda: self.client.shopping.hotel_offers_search.get(
                    cityCode=city_code,
                    checkInDate=check_in_date,
                    checkOutDate=check_out_date,
                    adults=adults,
                    roomQuantity=rooms
                ),
            ]
            
            for endpoint_func in endpoints_to_try:
                try:
                    response = endpoint_func()
                    if hasattr(response, 'data') and response.data:
                        hotel_offers = []
                        for hotel_data in response.data[:max_results]:
                            formatted_offer = self._format_hotel_offer(hotel_data)
                            hotel_offers.append(formatted_offer)
                        
                        return {
                            "success": True,
                            "results": hotel_offers,
                            "meta": {
                                "count": len(hotel_offers),
                                "approach": "direct_city_search"
                            }
                        }
                except Exception as e:
                    logger.warning(f"Direct city search variant failed: {str(e)}")
                    continue
            
            return {"success": False, "error": "Direct city search not available", "results": []}
            
        except Exception as e:
            return {"success": False, "error": f"Approach 1 failed: {str(e)}", "results": []}

    async def _search_hotels_approach_2(self, city_code: str, check_in_date: str, check_out_date: str,
                                    adults: int, rooms: int, max_results: int) -> Dict[str, Any]:
        """
        Approach 2: Traditional two-step approach with enhanced error handling and logging
        """
        try:
            logger.info(f"🔄 Approach 2: Two-step hotel search for {city_code}")
            
            # Step 1: Get hotel list by city
            logger.info(f"Step 1: Getting hotel list for city {city_code}")
            hotels_response = self.client.reference_data.locations.hotels.by_city.get(
                cityCode=city_code
            )
            
            if not hasattr(hotels_response, 'data') or not hotels_response.data:
                logger.warning(f"⚠️ Approach 2 Step 1 FAILED: No hotels found for city code {city_code}")
                return {
                    "success": False,
                    "error": f"No hotels found for city code: {city_code}",
                    "results": []
                }
            
            # Extract hotel IDs with better error handling
            hotel_ids = []
            for hotel in hotels_response.data[:max_results]:
                if isinstance(hotel, dict) and 'hotelId' in hotel:
                    hotel_ids.append(hotel['hotelId'])
                else:
                    logger.warning(f"Invalid hotel data structure: {hotel}")
            
            if not hotel_ids:
                logger.warning(f"⚠️ Approach 2 Step 1 FAILED: No valid hotel IDs found for city {city_code}")
                return {
                    "success": False,
                    "error": f"No valid hotel IDs found for city {city_code}",
                    "results": []
                }
            
            logger.info(f"Step 2: Searching offers for {len(hotel_ids)} hotels in {city_code}")
            
            # Step 2: Get hotel offers with enhanced parameter validation
            search_params = {
                'hotelIds': ','.join(hotel_ids[:20]),  # Limit to 20 hotels to avoid API limits
                'checkInDate': check_in_date,
                'checkOutDate': check_out_date,
                'adults': str(adults),  # Ensure string format
                'roomQuantity': str(rooms)  # Ensure string format
            }
            
            # Add logging for debugging
            logger.info(f"Hotel offers search params: {search_params}")
            
            offers_response = self.client.shopping.hotel_offers_search.get(**search_params)
            
            # Process results
            hotel_offers = []
            if hasattr(offers_response, 'data') and offers_response.data:
                for hotel_data in offers_response.data:
                    formatted_offer = self._format_hotel_offer(hotel_data)
                    hotel_offers.append(formatted_offer)
            
            logger.info(f"✅ Approach 2 SUCCESS: Found {len(hotel_offers)} hotel offers for {city_code}")
            
            return {
                "success": True,
                "results": hotel_offers,
                "meta": {
                    "count": len(hotel_offers),
                    "approach": "two_step_search",
                    "hotel_ids_found": len(hotel_ids),
                    "search_params": search_params
                }
            }
            
        except ResponseError as e:
            error_msg = f"Amadeus hotel search API error in approach 2: {str(e)}"
            logger.error(f"❌ Approach 2 AMADEUS ERROR for {city_code}: {error_msg}")
            
            # Extract more detailed error information
            status_code = 'unknown'
            error_details = str(e)
            if hasattr(e, 'response'):
                if hasattr(e.response, 'status_code'):
                    status_code = str(e.response.status_code)
                if hasattr(e.response, 'text'):
                    error_details = e.response.text
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": status_code,
                "error_details": error_details,
                "results": []
            }
            
        except Exception as e:
            logger.error(f"❌ Approach 2 EXCEPTION for {city_code}: {str(e)}")
            return {"success": False, "error": f"Approach 2 failed: {str(e)}", "results": []}

    async def _search_hotels_approach_3(self, city_code: str, check_in_date: str, check_out_date: str,
                                    adults: int, rooms: int, max_results: int) -> Dict[str, Any]:
        """
        Approach 3: Enhanced geocode-based hotel search with larger radius and more hotels
        """
        try:
            # ENHANCED: City code to coordinates mapping for major cities
            city_coordinates = {
                'NYC': {'latitude': 40.7128, 'longitude': -74.0060},
                'LON': {'latitude': 51.5074, 'longitude': -0.1278},
                'PAR': {'latitude': 48.8566, 'longitude': 2.3522},
                'TYO': {'latitude': 35.6762, 'longitude': 139.6503},
                'LAX': {'latitude': 34.0522, 'longitude': -118.2437},
                'SYD': {'latitude': -33.8688, 'longitude': 151.2093},
                'BKK': {'latitude': 13.7563, 'longitude': 100.5018},
                'SIN': {'latitude': 1.3521, 'longitude': 103.8198},
                'DXB': {'latitude': 25.2048, 'longitude': 55.2708},
                'HKG': {'latitude': 22.3193, 'longitude': 114.1694}
            }
            
            coords = city_coordinates.get(city_code)
            if not coords:
                return {"success": False, "error": f"No coordinates available for city {city_code}", "results": []}
            
            logger.info(f"Trying ENHANCED geocode search for {city_code} at {coords}")
            
            # FIXED: Use larger radius and get more hotels
            hotels_response = self.client.reference_data.locations.hotels.by_geocode.get(
                latitude=coords['latitude'],
                longitude=coords['longitude'],
                radius=100,  # INCREASED: 100km radius instead of 50km
                radiusUnit='KM'  # Explicit radius unit
            )
            
            if not hasattr(hotels_response, 'data') or not hotels_response.data:
                return {"success": False, "error": "No hotels found by geocode", "results": []}
            
            logger.info(f"Geocode search found {len(hotels_response.data)} hotels")
            
            # FIXED: Get more hotel IDs and process in batches if needed
            all_hotel_ids = [hotel['hotelId'] for hotel in hotels_response.data]
            logger.info(f"Available hotel IDs: {len(all_hotel_ids)}")
            
            # Process hotels in batches to avoid API limits
            all_hotel_offers = []
            batch_size = 20  # Process 20 hotels at a time
            
            for i in range(0, min(len(all_hotel_ids), max_results), batch_size):
                batch_hotel_ids = all_hotel_ids[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_hotel_ids)} hotels")
                
                try:
                    search_params = {
                        'hotelIds': ','.join(batch_hotel_ids),
                        'checkInDate': check_in_date,
                        'checkOutDate': check_out_date,
                        'adults': str(adults),
                        'roomQuantity': str(rooms)
                    }
                    
                    offers_response = self.client.shopping.hotel_offers_search.get(**search_params)
                    
                    if hasattr(offers_response, 'data') and offers_response.data:
                        for hotel_data in offers_response.data:
                            formatted_offer = self._format_hotel_offer(hotel_data)
                            all_hotel_offers.append(formatted_offer)
                            
                    logger.info(f"Batch {i//batch_size + 1} found {len(offers_response.data) if hasattr(offers_response, 'data') and offers_response.data else 0} offers")
                    
                except Exception as batch_error:
                    logger.warning(f"Batch {i//batch_size + 1} failed: {batch_error}")
                    continue
            
            logger.info(f"Total hotel offers found: {len(all_hotel_offers)}")
            
            return {
                "success": True,
                "results": all_hotel_offers,
                "meta": {
                    "count": len(all_hotel_offers),
                    "approach": "enhanced_geocode_search",
                    "coordinates": coords,
                    "radius": "100km",
                    "batches_processed": (min(len(all_hotel_ids), max_results) + batch_size - 1) // batch_size,
                    "total_hotels_available": len(all_hotel_ids)
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced approach 3 failed: {str(e)}")
            return {"success": False, "error": f"Enhanced approach 3 failed: {str(e)}", "results": []}      

    @amadeus_critical_retry
    async def get_hotel_offers(self, hotel_id: str, check_in_date: str, check_out_date: str,
                             adults: int = 1, rooms: int = 1) -> Dict[str, Any]:
        """
        Get specific hotel offers for a hotel with retry strategies
        """
        try:
            logger.info(f"Getting hotel offers for {hotel_id} with retry strategies")
            
            response = self.client.shopping.hotel_offers_search.get(
                hotelIds=hotel_id,
                checkInDate=check_in_date,
                checkOutDate=check_out_date,
                adults=adults,
                roomQuantity=rooms
            )
            
            if hasattr(response, 'data') and response.data:
                hotel_offers = [self._format_hotel_offer(hotel) for hotel in response.data]
                
                result = {
                    "success": True,
                    "offers": hotel_offers
                }
                
                # Validate response
                try:
                    if hotel_offers:
                        AmadeusHotelOffer(**hotel_offers[0])
                    logger.info(f"Hotel offers retrieved successfully: {len(hotel_offers)} offers")
                except ValidationError as ve:
                    logger.warning(f"Hotel offers validation failed: {ve}")
                    return {
                        "success": False,
                        "error": f"Offers validation failed: {str(ve)}"
                    }
                
                return result
            else:
                return {
                    "success": False,
                    "error": "No offers found for this hotel"
                }
                
        except ResponseError as e:
            error_msg = f"Amadeus hotel offers API error: {str(e)}"
            logger.error(error_msg)
            
            # Properly extract status code from Response object
            status_code = 'unknown'
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = str(e.response.status_code)
            
            return {
                "success": False,
                "error": error_msg,
                "error_code": status_code
            }
            
        except Exception as e:
            logger.error(f"Unexpected hotel offers error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    # ==================== SECONDARY OPERATIONS WITH RETRY STRATEGIES ====================

    @amadeus_secondary_retry
    async def search_airports(self, keyword: str, subtype: str = "AIRPORT") -> List[Dict[str, Any]]:
        """
        Search for airports using Amadeus Airport & City Search API with retry strategies
        """
        try:
            logger.info(f"Searching airports for '{keyword}' with retry strategies")
            
            response = self.client.reference_data.locations.get(
                keyword=keyword,
                subType=subtype
            )
            
            if hasattr(response, 'data') and response.data:
                logger.info(f"Airport search successful: {len(response.data)} results")
                return response.data
            else:
                logger.info(f"No airports found for keyword: {keyword}")
                return []
                
        except ResponseError as e:
            logger.error(f"Amadeus airport search API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected airport search error: {e}")
            return []

    @amadeus_secondary_retry
    async def get_flight_checkin_links(self, airline_code: str) -> List[Dict[str, Any]]:
        """
        Get flight check-in links for an airline with retry strategies
        """
        try:
            logger.info(f"Getting check-in links for airline {airline_code} with retry strategies")
            
            response = self.client.reference_data.urls.checkin_links.get(
                airlineCode=airline_code
            )
            
            if hasattr(response, 'data') and response.data:
                logger.info(f"Check-in links retrieved: {len(response.data)} links")
                return response.data
            else:
                logger.info(f"No check-in links found for airline: {airline_code}")
                return []
                
        except ResponseError as e:
            logger.error(f"Amadeus check-in links API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected check-in links error: {e}")
            return []

    # ==================== HELPER METHODS (UNCHANGED) ====================

    def _format_flight_offer(self, offer: Dict[str, Any]) -> Dict[str, Any]:
        """Format flight offer data for consistent response structure (unchanged)"""
        try:
            # Extract key information from the offer
            price = offer.get('price', {})
            itineraries = offer.get('itineraries', [])
            
            formatted_offer = {
                "id": offer.get('id'),
                "type": offer.get('type'),
                "source": offer.get('source'),
                "instantTicketingRequired": offer.get('instantTicketingRequired', False),
                "nonHomogeneous": offer.get('nonHomogeneous', False),
                "oneWay": offer.get('oneWay', False),
                "lastTicketingDate": offer.get('lastTicketingDate'),
                "numberOfBookableSeats": offer.get('numberOfBookableSeats'),
                "price": {
                    "currency": price.get('currency'),
                    "total": price.get('total'),
                    "base": price.get('base'),
                    "fees": price.get('fees', []),
                    "grandTotal": price.get('grandTotal')
                },
                "pricingOptions": offer.get('pricingOptions', {}),
                "validatingAirlineCodes": offer.get('validatingAirlineCodes', []),
                "travelerPricings": offer.get('travelerPricings', []),
                "itineraries": []
            }
            
            # Format itineraries
            for itinerary in itineraries:
                formatted_itinerary = {
                    "duration": itinerary.get('duration'),
                    "segments": []
                }
                
                for segment in itinerary.get('segments', []):
                    formatted_segment = {
                        "departure": segment.get('departure', {}),
                        "arrival": segment.get('arrival', {}),
                        "carrierCode": segment.get('carrierCode'),
                        "number": segment.get('number'),
                        "aircraft": segment.get('aircraft', {}),
                        "operating": segment.get('operating', {}),
                        "duration": segment.get('duration'),
                        "id": segment.get('id'),
                        "numberOfStops": segment.get('numberOfStops', 0),
                        "blacklistedInEU": segment.get('blacklistedInEU', False)
                    }
                    formatted_itinerary["segments"].append(formatted_segment)
                
                formatted_offer["itineraries"].append(formatted_itinerary)
            
            return formatted_offer
            
        except Exception as e:
            logger.error(f"Error formatting flight offer: {e}")
            return offer  # Return original if formatting fails

    def _format_hotel_offer(self, hotel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format hotel offer data for consistent response structure (unchanged)"""
        try:
            hotel = hotel_data.get('hotel', {})
            offers = hotel_data.get('offers', [])
            
            formatted_hotel = {
                "type": hotel_data.get('type'),
                "hotel": {
                    "chainCode": hotel.get('chainCode'),
                    "iataCode": hotel.get('iataCode'),
                    "dupeId": hotel.get('dupeId'),
                    "name": hotel.get('name'),
                    "hotelId": hotel.get('hotelId'),
                    "geoCode": hotel.get('geoCode', {}),
                    "address": hotel.get('address', {}),
                    "contact": hotel.get('contact', {}),
                    "amenities": hotel.get('amenities', []),
                    "rating": hotel.get('rating'),
                    "description": hotel.get('description', {})
                },
                "available": hotel_data.get('available', True),
                "offers": []
            }
            
            # Format offers
            for offer in offers:
                formatted_offer = {
                    "id": offer.get('id'),
                    "checkInDate": offer.get('checkInDate'),
                    "checkOutDate": offer.get('checkOutDate'),
                    "rateCode": offer.get('rateCode'),
                    "rateFamilyEstimated": offer.get('rateFamilyEstimated', {}),
                    "category": offer.get('category'),
                    "description": offer.get('description', {}),
                    "commission": offer.get('commission', {}),
                    "boardType": offer.get('boardType'),
                    "room": offer.get('room', {}),
                    "guests": offer.get('guests', {}),
                    "price": offer.get('price', {}),
                    "policies": offer.get('policies', {}),
                    "self": offer.get('self')
                }
                formatted_hotel["offers"].append(formatted_offer)
            
            return formatted_hotel
            
        except Exception as e:
            logger.error(f"Error formatting hotel offer: {e}")
            return hotel_data  # Return original if formatting fails
