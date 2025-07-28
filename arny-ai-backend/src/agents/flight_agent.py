"""
Flight Search Agent Module - ENHANCED VERSION with Cache Management

This module provides a flight search agent with enhanced capabilities including:
1. Support for up to 50 flight results from Amadeus API
2. Send all flights to OpenAI for filtering
3. Return up to 10 filtered flight results
4. Optimized for larger datasets
5. Cache management for improved performance

Usage example:
```python
from flight_agent import FlightAgent

# Create and use the agent
agent = FlightAgent()
result = await agent.process_message(user_id, "Find flights from Sydney to LA", session_id, {}, [])
```
"""

import uuid
import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from openai import OpenAI
from agents import Agent, function_tool, Runner, WebSearchTool

from ..utils.config import config
from ..services.amadeus_service import AmadeusService
from ..database.operations import DatabaseOperations
from ..database.models import FlightSearch
from .user_profile_agent import UserProfileAgent

# Configure logger
logger = logging.getLogger(__name__)

# Global variable to store the current agent instance
_current_flight_agent = None

def _get_flight_agent():
    """Get the current flight agent instance"""
    global _current_flight_agent
    return _current_flight_agent

def _run_async_safely(coro):
    """Run async coroutine safely by using the current event loop or creating a new one"""
    try:
        loop = asyncio.get_running_loop()
        # If there's already a running loop, we need to run in a thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_in_new_loop, coro)
            return future.result()
    except RuntimeError:
        # No running loop, we can use asyncio.run
        return asyncio.run(coro)

def _run_in_new_loop(coro):
    """Run coroutine in a completely new event loop"""
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()

# ==================== AIRPORT CODE MAPPING ====================

AIRPORT_CODE_MAPPING = {
    # ===== AUSTRALIA (Top 5) =====
    "sydney": "SYD",
    "melbourne": "MEL", 
    "brisbane": "BNE",
    "perth": "PER",
    "adelaide": "ADL",
    "darwin": "DRW",
    "canberra": "CBR",
    "gold coast": "OOL",
    "cairns": "CNS",
    "hobart": "HBA",

    # ===== NEW ZEALAND (Top 5) =====
    "auckland": "AKL",
    "wellington": "WLG",
    "christchurch": "CHC",
    "queenstown": "ZQN",
    "dunedin": "DUD",

    # ===== UNITED STATES (Top 20) =====
    "new york": "JFK",  # Primary NYC airport
    "los angeles": "LAX",
    "chicago": "ORD",
    "houston": "IAH",
    "phoenix": "PHX",
    "philadelphia": "PHL",
    "san antonio": "SAT",
    "san diego": "SAN",
    "dallas": "DFW",
    "san jose": "SJC",
    "austin": "AUS",
    "columbus": "CMH",
    "charlotte": "CLT",
    "san francisco": "SFO",
    "indianapolis": "IND",
    "seattle": "SEA",
    "denver": "DEN",
    "washington": "DCA",  # Primary Washington DC airport
    "washington dc": "DCA",
    "boston": "BOS",
    "detroit": "DTW",
    "nashville": "BNA",
    "portland": "PDX",
    "memphis": "MEM",
    "oklahoma city": "OKC",
    "las vegas": "LAS",
    "louisville": "SDF",
    "baltimore": "BWI",
    "milwaukee": "MKE",
    "albuquerque": "ABQ",
    "tucson": "TUS",
    "fresno": "FAT",
    "sacramento": "SMF",
    "kansas city": "MCI",
    "atlanta": "ATL",
    "colorado springs": "COS",
    "omaha": "OMA",
    "raleigh": "RDU",
    "miami": "MIA",
    "cleveland": "CLE",
    "tulsa": "TUL",
    "oakland": "OAK",
    "minneapolis": "MSP",
    "wichita": "ICT",

    # ===== UNITED KINGDOM (Top 5) =====
    "london": "LHR",  # Heathrow primary
    "london heathrow": "LHR",
    "london gatwick": "LGW",
    "london stansted": "STN",
    "london luton": "LTN",
    "manchester": "MAN",
    "birmingham": "BHX",
    "edinburgh": "EDI",
    "glasgow": "GLA",
    "bristol": "BRS",

    # ===== FRANCE (Top 5) =====
    "paris": "CDG",  # Charles de Gaulle primary
    "paris cdg": "CDG",
    "paris orly": "ORY",
    "lyon": "LYS",
    "marseille": "MRS",
    "nice": "NCE",
    "toulouse": "TLS",

    # ===== GERMANY (Top 5) =====
    "frankfurt": "FRA",
    "munich": "MUC",
    "berlin": "BER",
    "hamburg": "HAM",
    "dusseldorf": "DUS",
    "cologne": "CGN",
    "stuttgart": "STR",

    # ===== ITALY (Top 5) =====
    "rome": "FCO",  # Fiumicino primary
    "milan": "MXP",  # Malpensa primary
    "milan malpensa": "MXP",
    "milan linate": "LIN",
    "venice": "VCE",
    "naples": "NAP",
    "bologna": "BLQ",
    "genoa": "GOA",  # REQUESTED: Genoa, Italy
    "florence": "FLR",
    "turin": "TRN",

    # ===== SPAIN (Top 5) =====
    "madrid": "MAD",
    "barcelona": "BCN",
    "seville": "SVQ",
    "valencia": "VLC",
    "bilbao": "BIO",
    "palma": "PMI",
    "malaga": "AGP",

    # ===== NETHERLANDS (Top 5) =====
    "amsterdam": "AMS",
    "rotterdam": "RTM",
    "eindhoven": "EIN",
    "groningen": "GRQ",
    "maastricht": "MST",

    # ===== CANADA (Top 5) =====
    "toronto": "YYZ",
    "vancouver": "YVR",
    "montreal": "YUL",
    "calgary": "YYC",
    "ottawa": "YOW",
    "edmonton": "YEG",
    "winnipeg": "YWG",

    # ===== JAPAN (Top 5) =====
    "tokyo": "NRT",  # Narita primary
    "tokyo narita": "NRT",
    "tokyo haneda": "HND",
    "osaka": "KIX",  # Kansai primary
    "osaka kansai": "KIX",
    "osaka itami": "ITM",
    "nagoya": "NGO",
    "sapporo": "CTS",
    "fukuoka": "FUK",

    # ===== CHINA (Top 5) =====
    "beijing": "PEK",
    "shanghai": "PVG",  # Pudong primary
    "shanghai pudong": "PVG",
    "shanghai hongqiao": "SHA",
    "guangzhou": "CAN",
    "shenzhen": "SZX",
    "chengdu": "CTU",

    # ===== SOUTH KOREA (Top 5) =====
    "seoul": "ICN",  # Incheon primary
    "seoul incheon": "ICN",
    "seoul gimpo": "GMP",
    "busan": "PUS",
    "jeju": "CJU",
    "daegu": "TAE",

    # ===== INDIA (Top 5) =====
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "bangalore": "BLR",
    "chennai": "MAA",
    "hyderabad": "HYD",
    "kolkata": "CCU",
    "pune": "PNQ",

    # ===== SINGAPORE =====
    "singapore": "SIN",

    # ===== HONG KONG =====
    "hong kong": "HKG",

    # ===== UAE (Top 5) =====
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "sharjah": "SHJ",
    "ras al khaimah": "RKT",
    "fujairah": "FJR",

    # ===== THAILAND (Top 5) =====
    "bangkok": "BKK",  # Suvarnabhumi primary
    "bangkok suvarnabhumi": "BKK",
    "bangkok don mueang": "DMK",
    "phuket": "HKT",
    "chiang mai": "CNX",
    "koh samui": "USM",

    # ===== MALAYSIA (Top 5) =====
    "kuala lumpur": "KUL",
    "penang": "PEN",
    "kota kinabalu": "BKI",
    "kuching": "KCH",
    "langkawi": "LGK",

    # ===== INDONESIA (Top 5) =====
    "jakarta": "CGK",
    "bali": "DPS",  # Denpasar
    "denpasar": "DPS",
    "surabaya": "MLG",
    "medan": "KNO",
    "yogyakarta": "JOG",

    # ===== PHILIPPINES (Top 5) =====
    "manila": "MNL",
    "cebu": "CEB",
    "davao": "DVO",
    "clark": "CRK",
    "iloilo": "ILO",

    # ===== VIETNAM (Top 5) =====
    "ho chi minh city": "SGN",
    "saigon": "SGN",
    "hanoi": "HAN",
    "da nang": "DAD",
    "nha trang": "CXR",
    "phu quoc": "PQC",

    # ===== RUSSIA (Top 5) =====
    "moscow": "SVO",  # Sheremetyevo primary
    "moscow sheremetyevo": "SVO",
    "moscow domodedovo": "DME",
    "moscow vnukovo": "VKO",
    "st petersburg": "LED",
    "saint petersburg": "LED",
    "novosibirsk": "OVB",
    "yekaterinburg": "SVX",

    # ===== BRAZIL (Top 5) =====
    "sao paulo": "GRU",  # Guarulhos primary
    "sao paulo guarulhos": "GRU",
    "sao paulo congonhas": "CGH",
    "rio de janeiro": "GIG",  # Galeão primary
    "rio de janeiro galeao": "GIG",
    "rio de janeiro santos dumont": "SDU",
    "brasilia": "BSB",
    "salvador": "SSA",
    "fortaleza": "FOR",

    # ===== ARGENTINA (Top 5) =====
    "buenos aires": "EZE",  # Ezeiza primary
    "buenos aires ezeiza": "EZE",
    "buenos aires jorge newbery": "AEP",
    "cordoba": "COR",
    "mendoza": "MDZ",
    "bariloche": "BRC",

    # ===== MEXICO (Top 5) =====
    "mexico city": "MEX",
    "cancun": "CUN",
    "guadalajara": "GDL",
    "monterrey": "MTY",
    "puerto vallarta": "PVR",

    # ===== SOUTH AFRICA (Top 5) =====
    "cape town": "CPT",
    "johannesburg": "JNB",
    "durban": "DUR",
    "port elizabeth": "PLZ",
    "bloemfontein": "BFN",

    # ===== NIGERIA (Top 5) =====
    "lagos": "LOS",  # REQUESTED: Lagos, Nigeria
    "abuja": "ABV",
    "kano": "KAN",
    "port harcourt": "PHC",
    "enugu": "ENU",

    # ===== EGYPT (Top 5) =====
    "cairo": "CAI",
    "alexandria": "HBE",
    "hurghada": "HRG",
    "sharm el sheikh": "SSH",
    "luxor": "LXR",

    # ===== TURKEY (Top 5) =====
    "istanbul": "IST",  # Istanbul Airport primary
    "istanbul sabiha gokcen": "SAW",
    "ankara": "ESB",
    "antalya": "AYT",
    "izmir": "ADB",

    # ===== GREECE (Top 5) =====
    "athens": "ATH",
    "thessaloniki": "SKG",
    "heraklion": "HER",
    "rhodes": "RHO",
    "corfu": "CFU",

    # ===== ISRAEL (Top 5) =====
    "tel aviv": "TLV",
    "jerusalem": "JRS",
    "eilat": "ETH",
    "haifa": "HFA",
    "ovda": "VDA",

    # ===== SWITZERLAND (Top 5) =====
    "zurich": "ZUR",
    "geneva": "GVA",
    "basel": "BSL",
    "bern": "BRN",
    "lugano": "LUG",

    # ===== AUSTRIA (Top 5) =====
    "vienna": "VIE",
    "salzburg": "SZG",
    "innsbruck": "INN",
    "graz": "GRZ",
    "linz": "LNZ",

    # ===== BELGIUM (Top 5) =====
    "brussels": "BRU",
    "antwerp": "ANR",
    "liege": "LGG",
    "ostend": "OST",
    "charleroi": "CRL",

    # ===== DENMARK (Top 5) =====
    "copenhagen": "CPH",
    "billund": "BLL",
    "aalborg": "AAL",
    "aarhus": "AAR",
    "esbjerg": "EBJ",

    # ===== SWEDEN (Top 5) =====
    "stockholm": "ARN",  # Arlanda primary
    "stockholm arlanda": "ARN",
    "stockholm bromma": "BMA",
    "gothenburg": "GOT",
    "malmo": "MMX",
    "umea": "UME",

    # ===== NORWAY (Top 5) =====
    "oslo": "OSL",
    "bergen": "BGO",
    "trondheim": "TRD",
    "stavanger": "SVG",
    "tromso": "TOS",

    # ===== FINLAND (Top 5) =====
    "helsinki": "HEL",
    "tampere": "TMP",
    "turku": "TKU",
    "oulu": "OUL",
    "rovaniemi": "RVN",

    # ===== POLAND (Top 5) =====
    "warsaw": "WAW",
    "krakow": "KRK",
    "gdansk": "GDN",
    "wroclaw": "WRO",
    "poznan": "POZ",

    # ===== CZECH REPUBLIC (Top 5) =====
    "prague": "PRG",
    "brno": "BRQ",
    "ostrava": "OSR",
    "karlovy vary": "KLV",
    "pardubice": "PED",

    # ===== HUNGARY (Top 5) =====
    "budapest": "BUD",
    "debrecen": "DEB",
    "szeged": "QZD",
    "pecs": "QPJ",
    "miskolc": "MCQ",

    # ===== ROMANIA (Top 5) =====
    "bucharest": "OTP",  # Otopeni primary
    "cluj napoca": "CLJ",
    "timisoara": "TSR",
    "iasi": "IAS",
    "constanta": "CND",

    # ===== CROATIA (Top 5) =====
    "zagreb": "ZAG",
    "split": "SPU",
    "dubrovnik": "DBV",
    "pula": "PUY",
    "zadar": "ZAD",

    # ===== SERBIA (Top 5) =====
    "belgrade": "BEG",
    "nis": "INI",
    "novi sad": "QND",
    "kragujevac": "KGJ",
    "subotica": "QSU",

    # ===== MONTENEGRO (Top 5) =====
    "podgorica": "TGD",  # REQUESTED: Podgorica, Montenegro
    "tivat": "TIV",
    "niksic": "NIK",
    "bar": "BAR",
    "pljevlja": "PLJ",

    # ===== BULGARIA (Top 5) =====
    "sofia": "SOF",
    "plovdiv": "PDV",
    "varna": "VAR",
    "burgas": "BOJ",
    "ruse": "ROU",

    # ===== CHILE (Top 5) =====
    "santiago": "SCL",
    "valparaiso": "VAP",
    "concepcion": "CCP",
    "antofagasta": "ANF",
    "iquique": "IQQ",

    # ===== PERU (Top 5) =====
    "lima": "LIM",
    "cusco": "CUZ",
    "arequipa": "AQP",
    "trujillo": "TRU",
    "iquitos": "IQT",

    # ===== COLOMBIA (Top 5) =====
    "bogota": "BOG",
    "medellin": "MDE",
    "cartagena": "CTG",
    "cali": "CLO",
    "barranquilla": "BAQ",

    # ===== ECUADOR (Top 5) =====
    "quito": "UIO",
    "guayaquil": "GYE",
    "cuenca": "CUE",
    "manta": "MEC",
    "loja": "LOH",

    # ===== VENEZUELA (Top 5) =====
    "caracas": "CCS",
    "maracaibo": "MAR",
    "valencia": "VLN",
    "barquisimeto": "BRM",
    "puerto ordaz": "PZO",

    # Common city variations and aliases
    "nyc": "JFK",
    "la": "LAX",
    "sf": "SFO",
    "dc": "DCA",
    "chi": "ORD",
}

# ==================== STANDALONE TOOL FUNCTIONS ====================

@function_tool
def search_flights_tool(origin: str, destination: str, departure_date: str, 
                       return_date: Optional[str] = None, adults: int = 1, 
                       cabin_class: str = "ECONOMY") -> dict:
    """
    Search for flights using Amadeus API with group profile filtering and cache management
    
    Args:
        origin: Origin airport/city code (e.g., 'SYD', 'Sydney')
        destination: Destination airport/city code (e.g., 'LAX', 'Los Angeles')
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format (optional for one-way)
        adults: Number of adult passengers (default 1)
        cabin_class: Cabin class - ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
    
    Returns:
        Dict with flight search results and profile filtering information
    """
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        # ADDED: Cache management similar to hotel agent
        search_key = f"{origin}_{destination}_{departure_date}_{return_date}_{adults}_{cabin_class}"
        if hasattr(agent, '_search_cache') and search_key in agent._search_cache:
            print(f"⚡ CACHE HIT: Returning cached flight results for {search_key}")
            cached_result = agent._search_cache[search_key]
            
            # Update agent's latest search data with cached results
            agent.latest_search_results = cached_result.get("results", [])
            agent.latest_search_id = cached_result.get("search_id")
            agent.latest_filtering_info = cached_result.get("filtering_info", {})
            
            return cached_result
        
        print(f"✈️ Flight search: {origin} → {destination} on {departure_date}")
        
        # Convert city names to airport codes
        origin_code = AIRPORT_CODE_MAPPING.get(origin.lower(), origin.upper())
        destination_code = AIRPORT_CODE_MAPPING.get(destination.lower(), destination.upper())
        
        print(f"📍 Using airport codes: {origin_code} → {destination_code}")

        # Search flights using Amadeus
        search_params = {
            "origin": origin_code,
            "destination": destination_code,
            "departure_date": departure_date,
            "return_date": return_date,
            "adults": adults,
            "cabin_class": cabin_class
        }

        amadeus_response = _run_async_safely(agent.amadeus_service.search_flights(**search_params))

        # FIXED: Handle the response properly - extract results from the response
        if isinstance(amadeus_response, dict):
            if not amadeus_response.get("success", False):
                return {
                    "success": False,
                    "message": f"Flight search failed: {amadeus_response.get('error', 'Unknown error')}"
                }
            flights = amadeus_response.get("results", [])
        else:
            # Fallback for direct list response (shouldn't happen with current implementation)
            flights = amadeus_response if amadeus_response else []

        if not flights:
            return {
                "success": False,
                "message": f"No flights found for {origin_code} to {destination_code} on {departure_date}."
            }

        print(f"🔍 Found {len(flights)} flights from Amadeus API")

        # Create FlightSearch record
        search_id = str(uuid.uuid4())
        flight_search = FlightSearch(
            search_id=search_id,
            user_id=agent.current_user_id,
            session_id=agent.current_session_id,
            origin=origin_code,
            destination=destination_code,
            departure_date=departure_date,
            return_date=return_date,
            passengers=adults,  # FIXED: Use 'passengers' field name as defined in the model
            cabin_class=cabin_class,
            search_results=flights  # FIXED: Now passing the list of flights, not the entire response dict
        )
        
        # Save search to database (async)
        _run_async_safely(agent.db.save_flight_search(flight_search))

        # Apply profile filtering to flights
        original_count = len(flights)

        # FIXED: Call the correct method on UserProfileAgent with proper parameters
        filter_result = _run_async_safely(
            agent.profile_agent.filter_flight_results(
                user_id=agent.current_user_id,
                flight_results=flights,
                search_params=search_params
            )
        )

        # FIXED: Extract data from the returned dictionary structure
        filtered_flights = filter_result.get("filtered_results", flights[:10])
        filtering_applied = filter_result.get("filtering_applied", False)
        rationale = filter_result.get("reasoning", "Applied personalized filtering based on your profile")

        filtered_count = len(filtered_flights)
        
        print(f"🎯 Profile filtering: {original_count} → {filtered_count} flights")
        print(f"📊 Filtering applied: {filtering_applied}")
        if rationale:
            print(f"💡 Rationale: {rationale}")
        
        # Store results in agent for response
        agent.latest_search_results = filtered_flights
        agent.latest_search_id = search_id
        agent.latest_filtering_info = {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "filtering_applied": filtering_applied,
            "group_size": 1,  # Default for now
            "rationale": rationale
        }
        
        # Format results for presentation
        formatted_results = agent._format_flight_results_for_agent(
            filtered_flights, 
            origin_code, 
            destination_code, 
            departure_date,
            {
                "filtering_applied": filtering_applied,
                "original_count": original_count,
                "filtered_count": filtered_count,
                "group_size": 1,
                "rationale": rationale
            }
        )
        
        result_payload = {
            "success": True,
            "results": filtered_flights,
            "formatted_results": formatted_results,
            "search_id": flight_search.search_id,
            "search_params": search_params,
            "filtering_info": {
                "original_count": original_count,
                "filtered_count": filtered_count,
                "filtering_applied": filtering_applied,
                "group_size": 1,
                "rationale": rationale
            }
        }
        
        # ADDED: Cache the result (matching hotel agent pattern)
        if not hasattr(agent, '_search_cache'):
            agent._search_cache = {}
        agent._search_cache[search_key] = result_payload
        
        # ADDED: Keep cache manageable (same as hotel agent - max 10 entries)
        if len(agent._search_cache) > 10:
            oldest_key = list(agent._search_cache.keys())[0]
            del agent._search_cache[oldest_key]
        
        print(f"✅ Flight search completed successfully!")
        return result_payload
        
    except Exception as e:
        print(f"❌ Error in search_flights_tool: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error searching for flights: {str(e)}"
        }

@function_tool
def search_airports_tool(keyword: str, subtype: str = "AIRPORT") -> dict:
    """
    Search for airports based on keyword.
    
    Args:
        keyword: Search keyword, such as city name or airport code
        subtype: Location type, defaults to AIRPORT
    """
    logger.info(f"Searching airports: {keyword}, type: {subtype}")
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        airports = _run_async_safely(agent.amadeus_service.search_airports(keyword, subtype))
        
        if not airports:
            return {
                "success": False,
                "message": f"No airports found matching '{keyword}'."
            }
        
        result = f"Found {len(airports)} airports matching '{keyword}':\n\n"
        
        for airport in airports:
            name = airport.get('name', 'Unknown')
            iata_code = airport.get('iataCode', 'Unknown')
            city = airport.get('address', {}).get('cityName', 'Unknown')
            country = airport.get('address', {}).get('countryName', 'Unknown')
            
            result += f"- {name} ({iata_code})\n"
            result += f"  City: {city}, Country: {country}\n\n"
        
        return {
            "success": True,
            "message": result,
            "airports": airports
        }
    except Exception as e:
        logger.error(f"Error searching airports: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error searching for airports: {str(e)}"
        }

@function_tool
def get_checkin_links_tool(airline_code: str) -> dict:
    """
    Get online check-in links for an airline.
    
    Args:
        airline_code: Airline code (e.g., 'CA')
    """
    logger.info(f"Getting check-in links for airline: {airline_code}")
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        links = _run_async_safely(agent.amadeus_service.get_flight_checkin_links(airline_code))
        
        if not links:
            return {
                "success": False,
                "message": f"No check-in links found for airline {airline_code}."
            }
        
        result = f"Found {len(links)} check-in links for {airline_code} airline:\n\n"
        
        for link in links:
            link_type = link.get('type', 'Unknown')
            airline = link.get('airline', {}).get('name', airline_code)
            url = link.get('href', 'Unknown')
            
            result += f"- {airline}\n"
            result += f"  Type: {link_type}\n"
            result += f"  Link: {url}\n\n"
        
        return {
            "success": True,
            "message": result,
            "links": links
        }
        
    except Exception as e:
        logger.error(f"Error getting check-in links: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error getting check-in links: {str(e)}"
        }

@function_tool
def get_flight_pricing_tool(flight_selection: str) -> dict:
    """
    Get accurate pricing for a selected flight
    
    Args:
        flight_selection: Description of which flight the user selected (e.g., "flight 1", "the morning flight")
    
    Returns:
        Dict with detailed pricing information
    """
    
    try:
        agent = _get_flight_agent()
        if not agent:
            return {"success": False, "error": "Flight agent not available"}
        
        # Extract flight number from selection
        flight_number = agent._extract_flight_number(flight_selection)
        
        if not flight_number:
            return {
                "success": False,
                "message": "I couldn't determine which flight you selected. Please specify like 'flight 1' or 'the first option'."
            }
        
        # For now, return a detailed pricing response
        # In production, this would call Amadeus Flight Offers Price API
        return {
            "success": True,
            "message": f"Getting accurate pricing for flight {flight_number}...",
            "flight_number": flight_number,
            "pricing_info": {
                "note": "This would contain detailed pricing from Amadeus Flight Offers Price API",
                "includes": ["taxes", "fees", "cancellation_policy", "baggage_allowance"]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error getting flight pricing: {str(e)}"
        }

# ==================== FLIGHT AGENT CLASS ====================

class FlightAgent:
    """
    Flight agent using OpenAI Agents SDK with Amadeus API tools, profile filtering, and cache management
    """
    
    def __init__(self):
        global _current_flight_agent
        
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.amadeus_service = AmadeusService()
        self.db = DatabaseOperations()
        self.profile_agent = UserProfileAgent()
        
        # FIXED: Initialize context variables
        self.current_user_id = None
        self.current_session_id = None
        self.user_profile = None
        
        # FIXED: Add attributes to store latest search results for response
        self.latest_search_results = []
        self.latest_search_id = None
        self.latest_filtering_info = {}
        
        # ADDED: Initialize search cache for enhanced performance (matching hotel agent)
        self._search_cache = {}
        
        # Store this instance globally for tool access
        _current_flight_agent = self
        
        # Create the agent with enhanced flight search tools
        self.agent = Agent(
            name="Arny Flight Assistant",
            instructions=self._get_system_instructions(),
            model="o4-mini",
            tools=[
                search_flights_tool,
                search_airports_tool,
                get_checkin_links_tool,
                get_flight_pricing_tool,
                WebSearchTool()
            ]
        )
    
    async def _build_context_with_profile(self, user_profile: Dict[str, Any], 
                                        recent_conversation: List, max_messages: int = 50) -> List:
        """
        Build context that always includes user profile data plus recent conversation
        """
        
        # 1. Create profile context as system message
        profile_context = {
            "role": "system", 
            "content": f"""
            USER PROFILE CONTEXT:
            Name: {user_profile.get('name', 'Not provided')}
            Email: {user_profile.get('email', 'Not provided')}
            Gender: {user_profile.get('gender', 'Not provided')}
            Birthdate: {user_profile.get('birthdate', 'Not provided')}
            Location: {user_profile.get('city', 'Not provided')}
            Employer: {user_profile.get('employer', 'Not provided')}
            Working Schedule: {user_profile.get('working_schedule', 'Not provided')}
            Holiday Frequency: {user_profile.get('holiday_frequency', 'Not provided')}
            Annual Income: {user_profile.get('annual_income', 'Not provided')}
            Monthly Spending: {user_profile.get('monthly_spending', 'Not provided')}
            Holiday Preferences: {user_profile.get('holiday_preferences', [])}
            Group Code: {user_profile.get('group_code', 'None')}
            
            Use this profile information to personalize flight recommendations.
            Consider the user's budget (annual income and monthly spending), and preferences when suggesting flights.
            Take into account their holiday frequency to suggest appropriate booking timing.
            Always address the user by their name when possible.
            """
        }
        
        # 2. Get recent conversation (respecting limits)
        conversation_context = []
        for msg in recent_conversation[-max_messages:]:
            conversation_context.append({
                "role": msg.message_type,
                "content": msg.content
            })
        
        # 3. Combine: profile first, then conversation
        return [profile_context] + conversation_context
    
    async def process_message(self, user_id: str, message: str, session_id: str,
                            user_profile: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
        """
        Process flight search requests using OpenAI Agents SDK with profile filtering
        """
        
        try:
            print(f"✈️ FlightAgent processing message: '{message[:50]}...'")
            
            # FIXED: Clear previous search results at start of new message
            self.latest_search_results = []
            self.latest_search_id = None
            self.latest_filtering_info = {}
            
            # FIXED: Store context for tool calls on the current instance
            self.current_user_id = user_id
            self.current_session_id = session_id
            self.user_profile = user_profile
            
            # FIXED: Also update the global instance to ensure tools have access
            global _current_flight_agent
            _current_flight_agent.current_user_id = user_id
            _current_flight_agent.current_session_id = session_id
            _current_flight_agent.user_profile = user_profile
            # FIXED: Clear global instance results as well
            _current_flight_agent.latest_search_results = []
            _current_flight_agent.latest_search_id = None
            _current_flight_agent.latest_filtering_info = {}
            
            print(f"🔧 Context set: user_id={user_id}, session_id={session_id}")
            
            # Build context with profile + recent conversation
            full_context = await self._build_context_with_profile(user_profile, conversation_history)
            
            print(f"🔧 Processing with profile context + {len(conversation_history[-50:])} previous messages")
            
            # Process with agent
            if not conversation_history:
                # First message in conversation - profile context + new message
                print("🚀 Starting new flight conversation with profile context")
                result = await Runner.run(self.agent, full_context + [{"role": "user", "content": message}])
            else:
                # Continue conversation with profile + context + new message
                print("🔄 Continuing flight conversation with profile + context")
                result = await Runner.run(self.agent, full_context + [{"role": "user", "content": message}])
          
            # Extract response
            assistant_message = result.final_output
            
            print(f"✅ FlightAgent response generated: '{assistant_message[:50]}...'")
            
            # FIXED: Read search results from the global instance that tools updated
            global_agent = _get_flight_agent()
            search_results = global_agent.latest_search_results if global_agent else []
            search_id = global_agent.latest_search_id if global_agent else None
            filtering_info = global_agent.latest_filtering_info if global_agent else {}
            
            print(f"📊 Retrieved search data: {len(search_results)} results, search_id: {search_id}")
            
            # FIXED: Include search results and search ID in response from global instance
            return {
                "message": assistant_message,
                "agent_type": "flight",
                "requires_action": False,  # Will be set to True if flight selection is needed
                "search_results": search_results,
                "search_id": search_id,
                "filtering_info": filtering_info,
                "metadata": {
                    "agent_type": "flight",
                    "conversation_type": "flight_search"
                }
            }
        
        except Exception as e:
            print(f"❌ Error in flight agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "message": "I'm sorry, I encountered an error while searching for flights. Please try again.",
                "agent_type": "flight",
                "error": str(e),
                "requires_action": False,
                "search_results": [],
                "search_id": None,
                "filtering_info": {}
            }
    
    # ==================== HELPER METHODS ====================

    def _get_system_instructions(self) -> str:
        """Generate enhanced system instructions with current date and airport code mapping"""
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Format airport code mapping as string (showing first 20 examples)
        airport_mappings = "\n".join([f"- {city}: {code}" for city, code in list(AIRPORT_CODE_MAPPING.items())[:20]])
        
        return f"""You are Arny's professional flight search specialist. You help users find and book flights using the Amadeus flight search system with intelligent group filtering.     

Your main responsibilities are:
1. Understanding users' flight needs and extracting key information from natural language descriptions
2. Using the search_flights_tool to search for flights that meet user requirements
3. Using search_airports_tool when users need airport information or codes
4. Using get_checkin_links_tool when users ask about airline check-in
5. Using get_flight_pricing_tool when users want detailed pricing for specific flights

Current date: {today}
Tomorrow: {tomorrow}

Airport code examples:
{airport_mappings}

AMBIGUOUS CITY HANDLING WITH MOST RECENT CONTEXT PRIORITY:
- If a user requests flights to a city without specifying the country, first check the conversation history for the MOST RECENT previous flight or hotel search to that same city name
- Look through the conversation history from newest to oldest and find the most recent occurrence where the user specified the country for that city
- Use the country from that MOST RECENT occurrence for the current request
- If no previous context is found, or if the most recent previous context is also ambiguous, then ask the user to provide the country name first before proceeding with any search
- Once the country is determined (either from most recent context or user clarification), proceed with the appropriate flight search

Key rules:
1. ALWAYS use search_flights_tool for ANY flight search request
2. If no return date is provided, search for one-way flights
3. Default to 1 adult passenger unless specified otherwise
4. Default to ECONOMY class unless specified otherwise
5. Convert city names to airport codes automatically (e.g., Sydney → SYD, London → LHR)
6. Be specific about dates - ask for clarification if dates are unclear
7. **IMPORTANT: Present ALL filtered flight results in your response - do not truncate the list**
8. Show flight details including airlines, times, prices, and duration for ALL results
9. If multiple passengers, collect this information before searching

Example interactions:
- "Flights from Sydney to LA next Friday" → extract departure city, destination, and date
- "Round trip to Tokyo in March" → ask for specific dates and departure city
- "Business class to London" → search with travel_class="BUSINESS"

Always be helpful, professional, and efficient in finding the best flight options. When presenting search results, ensure you show ALL available flight options that were filtered for the user - never truncate or skip flights from your response."""

    def _format_flight_results_for_agent(self, flights: List[Dict], origin: str, destination: str, 
                                       departure_date: str, filtering_info: Dict) -> str:
        """Format flight results for the agent to present to the user"""
        
        if not flights:
            return "No flights found matching your criteria."
        
        result = f"Found {len(flights)} flights from {origin} to {destination} on {departure_date}:\n\n"
        
        if filtering_info.get("filtering_applied"):
            result += f"✨ Applied personalized filtering: {filtering_info.get('rationale', 'Based on your preferences')}\n"
            result += f"📊 Showing {filtering_info.get('filtered_count', len(flights))} of {filtering_info.get('original_count', len(flights))} available flights\n\n"
        
        for i, flight in enumerate(flights, 1):
            # Extract flight details
            price = flight.get('price', {})
            amount = price.get('total', 'N/A')
            currency = price.get('currency', '')
            
            # Get first itinerary and segment
            itineraries = flight.get('itineraries', [])
            if not itineraries:
                continue
                
            outbound = itineraries[0]
            segments = outbound.get('segments', [])
            if not segments:
                continue
                
            first_segment = segments[0]
            last_segment = segments[-1]
            
            # Flight details
            airline_code = first_segment.get('carrierCode', 'Unknown')
            flight_number = f"{airline_code}{first_segment.get('number', '')}"
            
            departure_time = first_segment.get('departure', {}).get('at', '')
            arrival_time = last_segment.get('arrival', {}).get('at', '')
            
            duration = outbound.get('duration', 'Unknown')
            stops = len(segments) - 1
            
            result += f"Flight {i}: {flight_number}\n"
            result += f"  💰 Price: {amount} {currency}\n"
            result += f"  🛫 Departure: {departure_time}\n"
            result += f"  🛬 Arrival: {arrival_time}\n"
            result += f"  ⏱️ Duration: {duration}\n"
            result += f"  🔄 Stops: {stops} {'stop' if stops == 1 else 'stops'}\n\n"
        
        return result

    def _extract_flight_number(self, selection: str) -> Optional[str]:
        """Extract flight number from user selection"""
        import re
        
        # Look for patterns like "flight 1", "option 2", "first", etc.
        patterns = [
            r'flight\s*(\d+)',
            r'option\s*(\d+)', 
            r'number\s*(\d+)',
            r'^(\d+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, selection.lower())
            if match:
                return match.group(1)
        
        # Handle word numbers
        word_numbers = {
            'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'
        }
        
        for word, num in word_numbers.items():
            if word in selection.lower():
                return num
        
        return None

# ==================== MODULE EXPORTS ====================

__all__ = [
    'FlightAgent',
    'search_flights_tool',
    'search_airports_tool', 
    'get_checkin_links_tool',
    'get_flight_pricing_tool'
]
