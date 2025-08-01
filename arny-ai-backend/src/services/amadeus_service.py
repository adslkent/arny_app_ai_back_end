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
        Approach 3: Enhanced geocode-based hotel search with comprehensive city coverage
        """
        try:
            # ENHANCED: City code to coordinates mapping for major cities
            city_coordinates = {
                # ===== EXISTING CITIES (KEEP THESE) =====
                'NYC': {'latitude': 40.7128, 'longitude': -74.0060},
                'LON': {'latitude': 51.5074, 'longitude': -0.1278},
                'PAR': {'latitude': 48.8566, 'longitude': 2.3522},
                'TYO': {'latitude': 35.6762, 'longitude': 139.6503},
                'LAX': {'latitude': 34.0522, 'longitude': -118.2437},
                'SYD': {'latitude': -33.8688, 'longitude': 151.2093},
                'BKK': {'latitude': 13.7563, 'longitude': 100.5018},
                'SIN': {'latitude': 1.3521, 'longitude': 103.8198},
                'DXB': {'latitude': 25.2048, 'longitude': 55.2708},
                'HKG': {'latitude': 22.3193, 'longitude': 114.1694},
                
                # ===== AUSTRALIA =====
                'MEL': {'latitude': -37.8136, 'longitude': 144.9631},
                'BNE': {'latitude': -27.4698, 'longitude': 153.0251},
                'PER': {'latitude': -31.9505, 'longitude': 115.8605},
                'ADL': {'latitude': -34.9285, 'longitude': 138.6007},
                'DRW': {'latitude': -12.4634, 'longitude': 130.8456},
                'CBR': {'latitude': -35.2809, 'longitude': 149.1300},
                'OOL': {'latitude': -28.1642, 'longitude': 153.5062},
                'CNS': {'latitude': -16.9186, 'longitude': 145.7781},
                'HBA': {'latitude': -42.8821, 'longitude': 147.3272},
                
                # ===== NEW ZEALAND =====
                'AKL': {'latitude': -36.8485, 'longitude': 174.7633},
                'WLG': {'latitude': -41.2865, 'longitude': 174.7762},
                'CHC': {'latitude': -43.5321, 'longitude': 172.6362},
                'ZQN': {'latitude': -45.0312, 'longitude': 168.6626},
                'DUD': {'latitude': -45.8788, 'longitude': 170.5028},
                
                # ===== UNITED STATES =====
                'CHI': {'latitude': 41.8781, 'longitude': -87.6298},
                'HOU': {'latitude': 29.7604, 'longitude': -95.3698},
                'PHX': {'latitude': 33.4484, 'longitude': -112.0740},
                'PHL': {'latitude': 39.9526, 'longitude': -75.1652},
                'SAT': {'latitude': 29.4241, 'longitude': -98.4936},
                'SAN': {'latitude': 32.7157, 'longitude': -117.1611},
                'DFW': {'latitude': 32.7767, 'longitude': -96.7970},
                'SJC': {'latitude': 37.3382, 'longitude': -121.8863},
                'AUS': {'latitude': 30.2672, 'longitude': -97.7431},
                'CMH': {'latitude': 39.9612, 'longitude': -82.9988},
                'CLT': {'latitude': 35.2271, 'longitude': -80.8431},
                'SFO': {'latitude': 37.7749, 'longitude': -122.4194},
                'IND': {'latitude': 39.7684, 'longitude': -86.1581},
                'SEA': {'latitude': 47.6062, 'longitude': -122.3321},
                'DEN': {'latitude': 39.7392, 'longitude': -104.9903},
                'WAS': {'latitude': 38.9072, 'longitude': -77.0369},
                'BOS': {'latitude': 42.3601, 'longitude': -71.0589},
                'DTT': {'latitude': 42.3314, 'longitude': -83.0458},
                'BNA': {'latitude': 36.1627, 'longitude': -86.7816},
                'PDX': {'latitude': 45.5152, 'longitude': -122.6784},
                'MEM': {'latitude': 35.1495, 'longitude': -90.0490},
                'OKC': {'latitude': 35.4676, 'longitude': -97.5164},
                'LAS': {'latitude': 36.1699, 'longitude': -115.1398},
                'SDF': {'latitude': 38.2527, 'longitude': -85.7585},
                'BWI': {'latitude': 39.2904, 'longitude': -76.6122},
                'MKE': {'latitude': 43.0389, 'longitude': -87.9065},
                'ABQ': {'latitude': 35.0844, 'longitude': -106.6504},
                'TUS': {'latitude': 32.2226, 'longitude': -110.9747},
                'FAT': {'latitude': 36.7378, 'longitude': -119.7871},
                'SMF': {'latitude': 38.5816, 'longitude': -121.4944},
                'MCI': {'latitude': 39.0997, 'longitude': -94.5786},
                'ATL': {'latitude': 33.7490, 'longitude': -84.3880},
                'COS': {'latitude': 38.8339, 'longitude': -104.8214},
                'OMA': {'latitude': 41.2524, 'longitude': -95.9980},
                'RDU': {'latitude': 35.7796, 'longitude': -78.6382},
                'MIA': {'latitude': 25.7617, 'longitude': -80.1918},
                'CLE': {'latitude': 41.4993, 'longitude': -81.6944},
                'TUL': {'latitude': 36.1540, 'longitude': -95.9928},
                'OAK': {'latitude': 37.8044, 'longitude': -122.2712},
                'MSP': {'latitude': 44.9778, 'longitude': -93.2650},
                'ICT': {'latitude': 37.6872, 'longitude': -97.3301},
                
                # ===== UNITED KINGDOM =====
                'MAN': {'latitude': 53.4808, 'longitude': -2.2426},
                'BHX': {'latitude': 52.4862, 'longitude': -1.8904},
                'EDI': {'latitude': 55.9533, 'longitude': -3.1883},
                'GLA': {'latitude': 55.8642, 'longitude': -4.2518},
                'BRS': {'latitude': 51.4545, 'longitude': -2.5879},
                'LPL': {'latitude': 53.4084, 'longitude': -2.9916},
                'LDS': {'latitude': 53.8008, 'longitude': -1.5491},
                'SHF': {'latitude': 53.3811, 'longitude': -1.4701},
                'NCL': {'latitude': 54.9783, 'longitude': -1.6178},
                
                # ===== FRANCE =====
                'LYS': {'latitude': 45.7640, 'longitude': 4.8357},
                'MRS': {'latitude': 43.2965, 'longitude': 5.3698},
                'NCE': {'latitude': 43.7102, 'longitude': 7.2620},
                'TLS': {'latitude': 43.6047, 'longitude': 1.4442},
                'SXB': {'latitude': 48.5734, 'longitude': 7.7521},
                'BOD': {'latitude': 44.8378, 'longitude': -0.5792},
                'LIL': {'latitude': 50.6292, 'longitude': 3.0573},
                'NTE': {'latitude': 47.2184, 'longitude': -1.5536},
                'MPL': {'latitude': 43.6108, 'longitude': 3.8767},
                
                # ===== GERMANY =====
                'BER': {'latitude': 52.5200, 'longitude': 13.4050},
                'MUC': {'latitude': 48.1351, 'longitude': 11.5820},
                'FRA': {'latitude': 50.1109, 'longitude': 8.6821},
                'HAM': {'latitude': 53.5511, 'longitude': 9.9937},
                'CGN': {'latitude': 50.9375, 'longitude': 6.9603},
                'STR': {'latitude': 48.7758, 'longitude': 9.1829},
                'DUS': {'latitude': 51.2277, 'longitude': 6.7735},
                'DTM': {'latitude': 51.5136, 'longitude': 7.4653},
                'ESS': {'latitude': 51.4556, 'longitude': 7.0116},
                'LEJ': {'latitude': 51.3397, 'longitude': 12.3731},
                
                # ===== ITALY =====
                'ROM': {'latitude': 41.9028, 'longitude': 12.4964},
                'MIL': {'latitude': 45.4642, 'longitude': 9.1900},
                'NAP': {'latitude': 40.8518, 'longitude': 14.2681},
                'TRN': {'latitude': 45.0703, 'longitude': 7.6869},
                'PMO': {'latitude': 38.1157, 'longitude': 13.3613},
                'GOA': {'latitude': 44.4056, 'longitude': 8.9463},
                'BLQ': {'latitude': 44.4949, 'longitude': 11.3426},
                'FLR': {'latitude': 43.7696, 'longitude': 11.2558},
                'BRI': {'latitude': 41.1171, 'longitude': 16.8719},
                'CTA': {'latitude': 37.5079, 'longitude': 15.0830},
                
                # ===== SPAIN =====
                'MAD': {'latitude': 40.4168, 'longitude': -3.7038},
                'BCN': {'latitude': 41.3851, 'longitude': 2.1734},
                'VLC': {'latitude': 39.4699, 'longitude': -0.3763},
                'SVQ': {'latitude': 37.3891, 'longitude': -5.9845},
                'ZAZ': {'latitude': 41.6488, 'longitude': -0.8891},
                'AGP': {'latitude': 36.7213, 'longitude': -4.4214},
                'MJV': {'latitude': 37.9922, 'longitude': -1.1307},
                'PMI': {'latitude': 39.5696, 'longitude': 2.6502},
                'LPA': {'latitude': 28.0916, 'longitude': -15.4509},
                'BIO': {'latitude': 43.2627, 'longitude': -2.9253},
                
                # ===== NETHERLANDS =====
                'AMS': {'latitude': 52.3676, 'longitude': 4.9041},
                'RTM': {'latitude': 51.9244, 'longitude': 4.4777},
                'HAG': {'latitude': 52.0705, 'longitude': 4.3007},
                'UTC': {'latitude': 52.0907, 'longitude': 5.1214},
                'EIN': {'latitude': 51.4416, 'longitude': 5.4697},
                
                # ===== CANADA =====
                'YTO': {'latitude': 43.6532, 'longitude': -79.3832},
                'YMQ': {'latitude': 45.5017, 'longitude': -73.5673},
                'YVR': {'latitude': 49.2827, 'longitude': -123.1207},
                'YYC': {'latitude': 51.0447, 'longitude': -114.0719},
                'YOW': {'latitude': 45.4215, 'longitude': -75.6972},
                'YEG': {'latitude': 53.5444, 'longitude': -113.4909},
                'YWG': {'latitude': 49.8951, 'longitude': -97.1384},
                'YQB': {'latitude': 46.8139, 'longitude': -71.2080},
                'YHM': {'latitude': 43.2557, 'longitude': -79.8711},
                'YXU': {'latitude': 43.0321, 'longitude': -81.1509},
                
                # ===== JAPAN =====
                'OSA': {'latitude': 34.6937, 'longitude': 135.5023},
                'KYO': {'latitude': 35.0116, 'longitude': 135.7681},
                'NGO': {'latitude': 35.1815, 'longitude': 136.9066},
                'SPK': {'latitude': 43.0642, 'longitude': 141.3469},
                'FUK': {'latitude': 33.5904, 'longitude': 130.4017},
                'SDJ': {'latitude': 38.2682, 'longitude': 140.8694},
                'HIJ': {'latitude': 34.3853, 'longitude': 132.4553},
                'OKA': {'latitude': 26.2123, 'longitude': 127.6792},
                'UKB': {'latitude': 34.6901, 'longitude': 135.1956},
                
                # ===== CHINA =====
                'BJS': {'latitude': 39.9042, 'longitude': 116.4074},
                'SHA': {'latitude': 31.2304, 'longitude': 121.4737},
                'CAN': {'latitude': 23.1291, 'longitude': 113.2644},
                'SZX': {'latitude': 22.5431, 'longitude': 114.0579},
                'CTU': {'latitude': 30.5728, 'longitude': 104.0668},
                'XIY': {'latitude': 34.3416, 'longitude': 108.9398},
                'HGH': {'latitude': 30.2741, 'longitude': 120.1551},
                'NKG': {'latitude': 32.0603, 'longitude': 118.7969},
                'WUH': {'latitude': 30.5928, 'longitude': 114.3055},
                'CKG': {'latitude': 29.5630, 'longitude': 106.5516},
                
                # ===== SOUTH KOREA =====
                'SEL': {'latitude': 37.5665, 'longitude': 126.9780},
                'PUS': {'latitude': 35.1796, 'longitude': 129.0756},
                'INC': {'latitude': 37.4563, 'longitude': 126.7052},
                'TAE': {'latitude': 35.8714, 'longitude': 128.6014},
                'KWJ': {'latitude': 35.1595, 'longitude': 126.8526},
                
                # ===== INDIA =====
                'BOM': {'latitude': 19.0760, 'longitude': 72.8777},
                'DEL': {'latitude': 28.7041, 'longitude': 77.1025},
                'BLR': {'latitude': 12.9716, 'longitude': 77.5946},
                'MAA': {'latitude': 13.0827, 'longitude': 80.2707},
                'HYD': {'latitude': 17.3850, 'longitude': 78.4867},
                'CCU': {'latitude': 22.5726, 'longitude': 88.3639},
                'PNQ': {'latitude': 18.5204, 'longitude': 73.8567},
                'AMD': {'latitude': 23.0225, 'longitude': 72.5714},
                'JAI': {'latitude': 26.9124, 'longitude': 75.7873},
                
                # ===== UAE =====
                'AUH': {'latitude': 24.4539, 'longitude': 54.3773},
                'SHJ': {'latitude': 25.3463, 'longitude': 55.4209},
                'AJM': {'latitude': 25.4052, 'longitude': 55.5136},
                'RKT': {'latitude': 25.7893, 'longitude': 55.9777},
                
                # ===== THAILAND =====
                'HKT': {'latitude': 7.8804, 'longitude': 98.3923},
                'CNX': {'latitude': 18.7883, 'longitude': 98.9853},
                'UTP': {'latitude': 12.9236, 'longitude': 100.8824},
                'USM': {'latitude': 9.5578, 'longitude': 100.0608},
                
                # ===== MALAYSIA =====
                'KUL': {'latitude': 3.1390, 'longitude': 101.6869},
                'PEN': {'latitude': 5.4164, 'longitude': 100.3327},
                'JHB': {'latitude': 1.4927, 'longitude': 103.7414},
                'BKI': {'latitude': 5.9749, 'longitude': 116.0724},
                'KCH': {'latitude': 1.5533, 'longitude': 110.3592},
                
                # ===== INDONESIA =====
                'JKT': {'latitude': -6.2088, 'longitude': 106.8456},
                'DPS': {'latitude': -8.3405, 'longitude': 115.0920},
                'MLG': {'latitude': -7.2575, 'longitude': 112.7521},
                'BDO': {'latitude': -6.9175, 'longitude': 107.6191},
                
                # ===== PHILIPPINES =====
                'MNL': {'latitude': 14.5995, 'longitude': 120.9842},
                'CEB': {'latitude': 10.3157, 'longitude': 123.8854},
                'DVO': {'latitude': 7.1907, 'longitude': 125.4553},
                'ILO': {'latitude': 10.7202, 'longitude': 122.5621},
                'CGY': {'latitude': 8.4542, 'longitude': 124.6319},
                
                # ===== VIETNAM =====
                'SGN': {'latitude': 10.8231, 'longitude': 106.6297},
                'HAN': {'latitude': 21.0285, 'longitude': 105.8542},
                'DAD': {'latitude': 16.0471, 'longitude': 108.2068},
                'HUI': {'latitude': 16.4637, 'longitude': 107.5909},
                
                # ===== RUSSIA =====
                'MOW': {'latitude': 55.7558, 'longitude': 37.6173},
                'LED': {'latitude': 59.9311, 'longitude': 30.3609},
                'OVB': {'latitude': 55.0084, 'longitude': 82.9357},
                'SVX': {'latitude': 56.8431, 'longitude': 60.6454},
                
                # ===== BRAZIL =====
                'SAO': {'latitude': -23.5505, 'longitude': -46.6333},
                'RIO': {'latitude': -22.9068, 'longitude': -43.1729},
                'BSB': {'latitude': -15.7975, 'longitude': -47.8919},
                'SSA': {'latitude': -12.9714, 'longitude': -38.5014},
                'FOR': {'latitude': -3.7319, 'longitude': -38.5267},
                'BHZ': {'latitude': -19.9167, 'longitude': -43.9345},
                'MAO': {'latitude': -3.1190, 'longitude': -60.0217},
                'CWB': {'latitude': -25.4284, 'longitude': -49.2733},
                'REC': {'latitude': -8.0476, 'longitude': -34.8770},
                'POA': {'latitude': -30.0346, 'longitude': -51.2177},
                
                # ===== ARGENTINA =====
                'BUE': {'latitude': -34.6118, 'longitude': -58.3960},
                'COR': {'latitude': -31.4201, 'longitude': -64.1888},
                'ROS': {'latitude': -32.9442, 'longitude': -60.6505},
                'MDZ': {'latitude': -32.8895, 'longitude': -68.8458},
                'LPL': {'latitude': -34.9215, 'longitude': -57.9545},
                
                # ===== MEXICO =====
                'MEX': {'latitude': 19.4326, 'longitude': -99.1332},
                'GDL': {'latitude': 20.6597, 'longitude': -103.3496},
                'MTY': {'latitude': 25.6866, 'longitude': -100.3161},
                'PBC': {'latitude': 19.0414, 'longitude': -98.2063},
                'TIJ': {'latitude': 32.5027, 'longitude': -117.0039},
                'BJX': {'latitude': 21.1212, 'longitude': -101.6835},
                'CJS': {'latitude': 31.6904, 'longitude': -106.4245},
                'TRC': {'latitude': 25.5428, 'longitude': -103.4068},
                'MID': {'latitude': 20.9674, 'longitude': -89.5926},
                'CUN': {'latitude': 21.1619, 'longitude': -86.8515},
                
                # ===== SOUTH AFRICA =====
                'JNB': {'latitude': -26.2041, 'longitude': 28.0473},
                'CPT': {'latitude': -33.9249, 'longitude': 18.4241},
                'DUR': {'latitude': -29.8587, 'longitude': 31.0218},
                'WDH': {'latitude': -22.5609, 'longitude': 17.0658},
                'PLZ': {'latitude': -33.9608, 'longitude': 25.6022},
                
                # ===== NIGERIA =====
                'LOS': {'latitude': 6.5244, 'longitude': 3.3792},
                'ABV': {'latitude': 9.0579, 'longitude': 7.4951},
                'KAN': {'latitude': 11.9756, 'longitude': 8.5264},
                'IBA': {'latitude': 7.3775, 'longitude': 3.9470},
                'PHC': {'latitude': 4.7514, 'longitude': 7.0128},
                
                # ===== EGYPT =====
                'CAI': {'latitude': 30.0444, 'longitude': 31.2357},
                'ALY': {'latitude': 31.2001, 'longitude': 29.9187},
                'SPX': {'latitude': 29.9773, 'longitude': 31.1325},
                'LXR': {'latitude': 25.6872, 'longitude': 32.6396},
                'ASW': {'latitude': 24.0889, 'longitude': 32.8998},
                
                # ===== TURKEY =====
                'IST': {'latitude': 41.0082, 'longitude': 28.9784},
                'ESB': {'latitude': 39.9334, 'longitude': 32.8597},
                'ADB': {'latitude': 38.4192, 'longitude': 27.1287},
                'AYT': {'latitude': 36.8969, 'longitude': 30.7133},
                'BTZ': {'latitude': 40.1826, 'longitude': 29.0669},
                
                # ===== GREECE =====
                'ATH': {'latitude': 37.9838, 'longitude': 23.7275},
                'SKG': {'latitude': 40.6401, 'longitude': 22.9444},
                'GPA': {'latitude': 38.2466, 'longitude': 21.7346},
                'HER': {'latitude': 35.3387, 'longitude': 25.1442},
                'LRA': {'latitude': 39.6390, 'longitude': 22.4194},
                
                # ===== ISRAEL =====
                'TLV': {'latitude': 32.0853, 'longitude': 34.7818},
                'JRS': {'latitude': 31.7683, 'longitude': 35.2137},
                'HFA': {'latitude': 32.7940, 'longitude': 34.9896},
                'BEV': {'latitude': 31.2530, 'longitude': 34.7915},
                'ASD': {'latitude': 31.8014, 'longitude': 34.6446},
                
                # ===== SWITZERLAND =====
                'ZUR': {'latitude': 47.3769, 'longitude': 8.5417},
                'GVA': {'latitude': 46.2044, 'longitude': 6.1432},
                'BSL': {'latitude': 47.5596, 'longitude': 7.5886},
                'BRN': {'latitude': 46.9481, 'longitude': 7.4474},
                'QLS': {'latitude': 46.5197, 'longitude': 6.6323},
                
                # ===== AUSTRIA =====
                'VIE': {'latitude': 48.2082, 'longitude': 16.3738},
                'SZG': {'latitude': 47.8095, 'longitude': 13.0550},
                'INN': {'latitude': 47.2692, 'longitude': 11.4041},
                'GRZ': {'latitude': 47.0707, 'longitude': 15.4395},
                'LNZ': {'latitude': 48.3069, 'longitude': 14.2858},
                
                # ===== BELGIUM =====
                'BRU': {'latitude': 50.8503, 'longitude': 4.3517},
                'ANR': {'latitude': 51.2194, 'longitude': 4.4025},
                'GNE': {'latitude': 51.0500, 'longitude': 3.7303},
                'CRL': {'latitude': 50.4108, 'longitude': 4.4446},
                'BRG': {'latitude': 51.2093, 'longitude': 3.2247},
                
                # ===== DENMARK =====
                'CPH': {'latitude': 55.6761, 'longitude': 12.5683},
                'AAR': {'latitude': 56.1629, 'longitude': 10.2039},
                'ODE': {'latitude': 55.4038, 'longitude': 10.4024},
                'AAL': {'latitude': 57.0488, 'longitude': 9.9217},
                'EBJ': {'latitude': 55.4760, 'longitude': 8.4380},
                
                # ===== SWEDEN =====
                'STO': {'latitude': 59.3293, 'longitude': 18.0686},
                'GOT': {'latitude': 57.7089, 'longitude': 11.9746},
                'MMX': {'latitude': 55.6050, 'longitude': 13.0038},
                'UPP': {'latitude': 59.8586, 'longitude': 17.6389},
                'VST': {'latitude': 59.6099, 'longitude': 16.5448},
                
                # ===== NORWAY =====
                'OSL': {'latitude': 59.9139, 'longitude': 10.7522},
                'BGO': {'latitude': 60.3913, 'longitude': 5.3221},
                'TRD': {'latitude': 63.4305, 'longitude': 10.3951},
                'SVG': {'latitude': 58.9700, 'longitude': 5.7331},
                'KRS': {'latitude': 58.1467, 'longitude': 7.9956},
                
                # ===== FINLAND =====
                'HEL': {'latitude': 60.1699, 'longitude': 24.9384},
                'ESP': {'latitude': 60.2055, 'longitude': 24.6559},
                'TMP': {'latitude': 61.4991, 'longitude': 23.7871},
                'VAN': {'latitude': 60.2934, 'longitude': 25.0408},
                'TKU': {'latitude': 60.4518, 'longitude': 22.2666},
                
                # ===== POLAND =====
                'WAW': {'latitude': 52.2297, 'longitude': 21.0122},
                'KRK': {'latitude': 50.0647, 'longitude': 19.9450},
                'LCJ': {'latitude': 51.7592, 'longitude': 19.4560},
                'WRO': {'latitude': 51.1079, 'longitude': 17.0385},
                'POZ': {'latitude': 52.4064, 'longitude': 16.9252},
                
                # ===== CZECH REPUBLIC =====
                'PRG': {'latitude': 50.0755, 'longitude': 14.4378},
                'BRQ': {'latitude': 49.1951, 'longitude': 16.6068},
                'OSR': {'latitude': 49.8209, 'longitude': 18.2625},
                'PLZ': {'latitude': 49.7384, 'longitude': 13.3736},
                'LBC': {'latitude': 50.7663, 'longitude': 15.0543},
                
                # ===== HUNGARY =====
                'BUD': {'latitude': 47.4979, 'longitude': 19.0402},
                'DEB': {'latitude': 47.5316, 'longitude': 21.6273},
                'SZD': {'latitude': 46.2530, 'longitude': 20.1414},
                'MCQ': {'latitude': 48.1034, 'longitude': 20.7784},
                'PCS': {'latitude': 46.0727, 'longitude': 18.2330},
                
                # ===== ROMANIA =====
                'BUH': {'latitude': 44.4268, 'longitude': 26.1025},
                'CLJ': {'latitude': 46.7712, 'longitude': 23.6236},
                'TSR': {'latitude': 45.7489, 'longitude': 21.2087},
                'IAS': {'latitude': 47.1585, 'longitude': 27.6014},
                'CND': {'latitude': 44.1598, 'longitude': 28.6348},
                
                # ===== CROATIA =====
                'ZAG': {'latitude': 45.8150, 'longitude': 15.9819},
                'SPU': {'latitude': 43.5081, 'longitude': 16.4402},
                'RJK': {'latitude': 45.3271, 'longitude': 14.4422},
                'OSI': {'latitude': 45.5550, 'longitude': 18.6955},
                'ZAD': {'latitude': 44.1194, 'longitude': 15.2314},
                
                # ===== SERBIA =====
                'BEG': {'latitude': 44.7866, 'longitude': 20.4489},
                'QND': {'latitude': 45.2671, 'longitude': 19.8335},
                'INI': {'latitude': 43.3209, 'longitude': 21.8958},
                'KGJ': {'latitude': 44.0122, 'longitude': 20.9111},
                'QSU': {'latitude': 46.1717, 'longitude': 19.6669},
                
                # ===== MONTENEGRO =====
                'TGD': {'latitude': 42.4304, 'longitude': 19.2594},
                'TIV': {'latitude': 42.4047, 'longitude': 18.7230},
                'BDV': {'latitude': 42.2864, 'longitude': 18.8400},
                'BAR': {'latitude': 42.0941, 'longitude': 19.0905},
                'NIK': {'latitude': 42.7731, 'longitude': 18.9447},
                
                # ===== BULGARIA =====
                'SOF': {'latitude': 42.6977, 'longitude': 23.3219},
                'PDV': {'latitude': 42.1354, 'longitude': 24.7453},
                'VAR': {'latitude': 43.2141, 'longitude': 27.9147},
                'BOJ': {'latitude': 42.4939, 'longitude': 27.4721},
                'ROU': {'latitude': 43.8563, 'longitude': 25.9700},
                
                # ===== CHILE =====
                'SCL': {'latitude': -33.4489, 'longitude': -70.6693},
                'VAP': {'latitude': -33.0472, 'longitude': -71.6127},
                'CCP': {'latitude': -36.8201, 'longitude': -73.0444},
                'LSC': {'latitude': -29.9027, 'longitude': -71.2519},
                'ANF': {'latitude': -23.6509, 'longitude': -70.3975},
                
                # ===== PERU =====
                'LIM': {'latitude': -12.0464, 'longitude': -77.0428},
                'AQP': {'latitude': -16.4090, 'longitude': -71.5375},
                'TRU': {'latitude': -8.1116, 'longitude': -79.0287},
                'CIX': {'latitude': -6.7714, 'longitude': -79.8441},
                'PIU': {'latitude': -5.1945, 'longitude': -80.6328},
                
                # ===== COLOMBIA =====
                'BOG': {'latitude': 4.7110, 'longitude': -74.0721},
                'MDE': {'latitude': 6.2442, 'longitude': -75.5812},
                'CLO': {'latitude': 3.4516, 'longitude': -76.5320},
                'BAQ': {'latitude': 10.9639, 'longitude': -74.7964},
                'CTG': {'latitude': 10.3910, 'longitude': -75.4794},
                
                # ===== ECUADOR =====
                'UIO': {'latitude': -0.1807, 'longitude': -78.4678},
                'GYE': {'latitude': -2.1894, 'longitude': -79.8890},
                'CUE': {'latitude': -2.9001, 'longitude': -79.0059},
                'STD': {'latitude': -0.2500, 'longitude': -79.1750},
                'MHC': {'latitude': -3.2581, 'longitude': -79.9553},
                
                # ===== VENEZUELA =====
                'CCS': {'latitude': 10.4806, 'longitude': -66.9036},
                'MAR': {'latitude': 10.6427, 'longitude': -71.6125},
                'VLN': {'latitude': 10.1621, 'longitude': -68.0077},
                'BRM': {'latitude': 10.0647, 'longitude': -69.3570},
                'CGU': {'latitude': 8.3114, 'longitude': -62.7116},
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
