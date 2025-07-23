import re
import json
import base64
import requests
import email.mime.text
import email.mime.multipart
import os
import logging
from typing import List, Dict, Any, Optional
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import msal
from openai import OpenAI
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

def is_lambda_environment() -> bool:
    """Detect if running in AWS Lambda environment"""
    return (
        os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None or
        os.environ.get('LAMBDA_RUNTIME_DIR') is not None or
        os.environ.get('AWS_EXECUTION_ENV') is not None
    )

# ==================== PYDANTIC MODELS FOR VALIDATION ====================

class GooglePersonName(BaseModel):
    """Pydantic model for Google People API name validation"""
    displayName: Optional[str] = None
    familyName: Optional[str] = None
    givenName: Optional[str] = None

class GooglePersonBirthday(BaseModel):
    """Pydantic model for Google People API birthday validation"""
    date: Optional[Dict[str, int]] = None

class GooglePersonGender(BaseModel):
    """Pydantic model for Google People API gender validation"""
    value: Optional[str] = None

class GooglePersonAddress(BaseModel):
    """Pydantic model for Google People API address validation"""
    value: Optional[str] = None
    type: Optional[str] = None

class GooglePersonResponse(BaseModel):
    """Pydantic model for Google People API response validation"""
    names: Optional[List[GooglePersonName]] = None
    birthdays: Optional[List[GooglePersonBirthday]] = None
    genders: Optional[List[GooglePersonGender]] = None
    addresses: Optional[List[GooglePersonAddress]] = None

class EmailScanResult(BaseModel):
    """Pydantic model for email scan result validation"""
    name: Optional[str] = None
    gender: Optional[str] = None
    birthdate: Optional[str] = None
    city: Optional[str] = None
    success: bool
    error: Optional[str] = None

class OpenAIResponse(BaseModel):
    """Pydantic model for OpenAI response validation"""
    output: Optional[List[Any]] = None

# ==================== OPENAI API RETRY CONDITIONS ====================

def retry_on_openai_api_error_result(result):
    """Condition 2: Retry if OpenAI API result contains error/warning fields"""
    if hasattr(result, 'error') and result.error:
        return True
    if isinstance(result, dict):
        return (
            result.get("error") is not None or
            "warning" in result or
            result.get("success") is False
        )
    return False

def retry_on_openai_http_status_error(result):
    """Condition 1: Retry on HTTP status errors for OpenAI API calls"""
    if hasattr(result, 'status_code'):
        return result.status_code >= 400
    if hasattr(result, 'response') and hasattr(result.response, 'status_code'):
        return result.response.status_code >= 400
    return False

def retry_on_openai_validation_failure(result):
    """Condition 5: Retry if OpenAI result fails Pydantic validation"""
    try:
        if result:
            OpenAIResponse.model_validate(result.__dict__ if hasattr(result, '__dict__') else result)
        return False
    except (ValidationError, AttributeError):
        return True

def retry_on_openai_api_exception(exception):
    """Condition 4: Custom exception checker for OpenAI API calls"""
    exception_str = str(exception).lower()
    return any(keyword in exception_str for keyword in [
        'timeout', 'failed', 'unavailable', 'rate limit', 'api error',
        'connection', 'network', 'server error'
    ])

# OpenAI API retry decorator with all 5 conditions
openai_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|rate.limit|api.error|connection|network|server.error).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_openai_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_openai_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_openai_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_openai_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# ==================== EXISTING RETRY CONDITIONS FOR OTHER APIS ====================

def retry_on_google_api_error_result(result):
    """Retry if Google API result contains errors or warnings"""
    if hasattr(result, 'error') and result.error:
        return True
    if isinstance(result, dict):
        return (
            result.get("error") is not None or
            "warning" in result or
            result.get("success") is False
        )
    return False

def retry_on_http_status_error(result):
    """Retry on HTTP status errors"""
    if hasattr(result, 'status_code'):
        return result.status_code >= 400
    if hasattr(result, 'response') and hasattr(result.response, 'status_code'):
        return result.response.status_code >= 400
    return False

def retry_on_api_validation_failure(result):
    """Retry if result fails validation"""
    try:
        if result and isinstance(result, dict):
            EmailScanResult(**result)
        return False
    except (ValidationError, TypeError):
        return True

def retry_on_google_api_exception(exception):
    """Custom exception checker for Google API calls"""
    exception_str = str(exception).lower()
    return any(keyword in exception_str for keyword in [
        'timeout', 'failed', 'unavailable', 'quota exceeded', 'invalid_grant',
        'insufficient permissions', 'rate limit'
    ])

def retry_on_microsoft_api_exception(exception):
    """Custom exception checker for Microsoft API calls"""
    exception_str = str(exception).lower()
    return any(keyword in exception_str for keyword in [
        'timeout', 'failed', 'unavailable', 'invalid_grant', 'consent_required',
        'token_expired', 'interaction_required'
    ])

# Google API retry decorator
google_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|quota|invalid_grant|rate.limit).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_google_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_google_api_error_result),
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_api_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

# Microsoft API retry decorator
microsoft_api_retry = retry(
    retry=retry_any(
        # Condition 3: Exception message matching
        retry_if_exception_message(match=r".*(timeout|failed|unavailable|invalid_grant|consent_required|token_expired).*"),
        # Condition 4: Exception types and custom checkers
        retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError, requests.exceptions.Timeout)),
        retry_if_exception(retry_on_microsoft_api_exception),
        # Condition 2: Error/warning field inspection
        retry_if_result(retry_on_google_api_error_result),  # Same logic applies
        # Condition 1: HTTP status code checking
        retry_if_result(retry_on_http_status_error),
        # Condition 5: Validation failure
        retry_if_result(retry_on_api_validation_failure)
    ),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=15),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)

class EmailService:
    """Enhanced service for email operations with Lambda-compatible OAuth and Tenacity retry strategies"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.stored_credentials = {}  # Store user credentials by user_id
        self.is_lambda = is_lambda_environment()
        
        print(f"🔧 EmailService initialized - Lambda environment: {self.is_lambda}")
    
    @openai_api_retry
    def extract_city_with_ai(self, address_text: str) -> Optional[str]:
        """Use OpenAI to extract city from address text with Tenacity retry strategies"""
        if not address_text or len(address_text) < 2:
            return None
        
        try:
            print(f"🧠 AI extracting city from: '{address_text}'")
            
            input_prompt = f"""Extract ONLY the city name from this address: "{address_text}"

Rules:
- Return ONLY the city name, nothing else
- Do not include state, country, postal codes, or street details
- For international addresses, return the city in English if possible
- If multiple cities are mentioned, return the main/primary city
- If no clear city can be identified, return "UNKNOWN"

Examples:
- "803/27 King Street, Sydney NSW 2000" → "Sydney"
- "123 Main St, New York, NY 10001" → "New York"
- "45 Avenue des Champs-Élysées, 75008 Paris, France" → "Paris"
- "1-1-1 Shibuya, Shibuya City, Tokyo 150-0002, Japan" → "Tokyo"
- "Unter den Linden 1, 10117 Berlin, Germany" → "Berlin"
- "Microsoft Corporation, Redmond, WA" → "Redmond"

Address: "{address_text}"
City:"""

            response = self.openai_client.responses.create(
                model="o4-mini",
                input=input_prompt
            )
            
            if response and hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text') and content_item.text:
                                city = content_item.text.strip()
                                
                                if city and city != "UNKNOWN" and len(city) <= 50 and not any(char.isdigit() for char in city):
                                    city = city.strip('"\'')
                                    print(f"✅ AI extracted city: '{city}' from '{address_text}'")
                                    return city
                                else:
                                    print(f"❌ AI returned invalid city: '{city}' from '{address_text}'")
            
            print(f"❌ AI could not extract city from: '{address_text}'")
            return None
            
        except Exception as e:
            print(f"❌ City extraction failed for '{address_text}': {str(e)}")
            return None
    
    @google_api_retry
    def scan_gmail_profile_server_to_server(self, email: str, user_id: str) -> Dict[str, Any]:
        """
        Server-to-server Gmail profile scanning using service account with Tenacity retry strategies
        NOTE: This requires domain-wide delegation setup for business emails
        """
        try:
            print(f"📧 Attempting server-to-server Gmail profile scan for: {email}")
            
            # Check if we have service account credentials
            service_account_file = os.environ.get('GOOGLE_SERVICE_ACCOUNT_FILE')
            if not service_account_file or not os.path.exists(service_account_file):
                print("❌ No Google service account file found")
                result = {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Service account credentials not configured"
                }
                
                # Validate result before returning
                try:
                    EmailScanResult(**result)
                except ValidationError as ve:
                    logger.warning(f"Gmail scan result validation failed: {ve}")
                    return {
                        "name": None,
                        "gender": None,
                        "birthdate": None,
                        "city": None,
                        "success": False,
                        "error": "Result validation failed"
                    }
                
                return result
            
            # Load service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/userinfo.profile']
            )
            
            # Delegate to user email for impersonation
            delegated_credentials = credentials.with_subject(email)
            
            # Build People API service
            people_service = build('people', 'v1', credentials=delegated_credentials)
            
            # Get profile information
            profile = people_service.people().get(
                resourceName='people/me',
                personFields='names,birthdays,genders,addresses'
            ).execute()
            
            print(f"📊 Retrieved profile data for {email}")
            
            # Validate profile response
            try:
                validated_profile = GooglePersonResponse(**profile)
            except ValidationError as ve:
                logger.warning(f"Profile response validation failed: {ve}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Profile validation failed"
                }
            
            # Extract information
            name = None
            gender = None
            birthdate = None
            city = None
            
            # Extract name
            if validated_profile.names and len(validated_profile.names) > 0:
                name_data = validated_profile.names[0]
                name = name_data.displayName or f"{name_data.givenName or ''} {name_data.familyName or ''}".strip()
            
            # Extract gender
            if validated_profile.genders and len(validated_profile.genders) > 0:
                gender = validated_profile.genders[0].value
            
            # Extract birthdate
            if validated_profile.birthdays and len(validated_profile.birthdays) > 0:
                birthday_data = validated_profile.birthdays[0]
                if birthday_data.date:
                    date_info = birthday_data.date
                    year = date_info.get('year')
                    month = date_info.get('month')
                    day = date_info.get('day')
                    
                    if year and month and day:
                        birthdate = f"{year:04d}-{month:02d}-{day:02d}"
            
            # Extract city from addresses
            if validated_profile.addresses and len(validated_profile.addresses) > 0:
                for address_data in validated_profile.addresses:
                    if address_data.value and address_data.type in ['home', 'work']:
                        city = self.extract_city_with_ai(address_data.value)
                        if city:
                            break
            
            result = {
                "name": name,
                "gender": gender,
                "birthdate": birthdate,
                "city": city,
                "success": True,
                "error": None
            }
            
            # Validate final result
            try:
                EmailScanResult(**result)
            except ValidationError as ve:
                logger.warning(f"Final result validation failed: {ve}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Final result validation failed"
                }
            
            print(f"✅ Successfully scanned Gmail profile for {email}")
            return result
            
        except Exception as e:
            print(f"❌ Gmail server-to-server scan failed for {email}: {str(e)}")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": f"Gmail scan failed: {str(e)}"
            }
    
    @microsoft_api_retry
    def scan_outlook_profile_server_to_server(self, email: str, user_id: str) -> Dict[str, Any]:
        """
        Server-to-server Outlook profile scanning using app-only authentication with Tenacity retry strategies
        NOTE: This requires admin consent for the Microsoft Graph API permissions
        """
        try:
            print(f"📧 Attempting server-to-server Outlook profile scan for: {email}")
            
            # Check if we have Azure AD app credentials
            client_id = os.environ.get('MICROSOFT_CLIENT_ID')
            client_secret = os.environ.get('MICROSOFT_CLIENT_SECRET')
            tenant_id = os.environ.get('MICROSOFT_TENANT_ID')
            
            if not all([client_id, client_secret, tenant_id]):
                print("❌ Microsoft Azure AD app credentials not configured")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Microsoft credentials not configured"
                }
            
            # Create MSAL app instance
            app = msal.ConfidentialClientApplication(
                client_id=client_id,
                client_credential=client_secret,
                authority=f"https://login.microsoftonline.com/{tenant_id}"
            )
            
            # Get app-only access token
            token_result = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )
            
            if 'access_token' not in token_result:
                print(f"❌ Failed to acquire access token: {token_result.get('error_description', 'Unknown error')}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Failed to acquire access token"
                }
            
            access_token = token_result['access_token']
            
            # Make Graph API call to get user profile
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Get user profile information
            response = requests.get(
                f'https://graph.microsoft.com/v1.0/users/{email}',
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to get user profile: {response.status_code} - {response.text}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": f"Graph API error: {response.status_code}"
                }
            
            profile_data = response.json()
            print(f"📊 Retrieved Outlook profile data for {email}")
            
            # Extract information
            name = profile_data.get('displayName')
            city = profile_data.get('city')
            
            # Try to get additional information from user's contact
            contact_response = requests.get(
                f'https://graph.microsoft.com/v1.0/users/{email}/contacts',
                headers=headers,
                timeout=10
            )
            
            birthdate = None
            gender = None
            
            if contact_response.status_code == 200:
                contacts_data = contact_response.json()
                if 'value' in contacts_data and len(contacts_data['value']) > 0:
                    # Look for self-contact with birthday info
                    for contact in contacts_data['value']:
                        if contact.get('emailAddresses'):
                            for email_addr in contact['emailAddresses']:
                                if email_addr.get('address', '').lower() == email.lower():
                                    birthdate = contact.get('birthday')
                                    gender = contact.get('personalNotes', '').lower() if 'gender' in contact.get('personalNotes', '').lower() else None
                                    break
            
            # If we still don't have city, try to extract from office location
            if not city and profile_data.get('officeLocation'):
                city = self.extract_city_with_ai(profile_data['officeLocation'])
            
            result = {
                "name": name,
                "gender": gender,
                "birthdate": birthdate,
                "city": city,
                "success": True,
                "error": None
            }
            
            # Validate final result
            try:
                EmailScanResult(**result)
            except ValidationError as ve:
                logger.warning(f"Outlook scan result validation failed: {ve}")
                return {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": False,
                    "error": "Result validation failed"
                }
            
            print(f"✅ Successfully scanned Outlook profile for {email}")
            return result
            
        except Exception as e:
            print(f"❌ Outlook server-to-server scan failed for {email}: {str(e)}")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": f"Outlook scan failed: {str(e)}"
            }
    
    def scan_email_for_profile(self, email: str, user_id: str) -> Dict[str, Any]:
        """
        Scan email for profile information using environment-appropriate method
        """
        try:
            print(f"📧 Starting email profile scan for: {email}")
            
            # Determine email provider
            email_domain = email.split('@')[-1].lower()
            
            if email_domain in ['gmail.com', 'googlemail.com']:
                print(f"🔍 Detected Gmail account: {email}")
                return self.scan_gmail_profile_server_to_server(email, user_id)
            
            elif email_domain in ['outlook.com', 'hotmail.com', 'live.com', 'msn.com']:
                print(f"🔍 Detected Microsoft account: {email}")
                return self.scan_outlook_profile_server_to_server(email, user_id)
            
            else:
                print(f"🔍 Unknown email provider for: {email}")
                # For unknown providers, return empty but successful result
                result = {
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "city": None,
                    "success": True,
                    "error": "Email provider not supported for automatic scanning"
                }
                
                # Validate result
                try:
                    EmailScanResult(**result)
                except ValidationError:
                    return {
                        "name": None,
                        "gender": None,
                        "birthdate": None,
                        "city": None,
                        "success": False,
                        "error": "Result validation failed"
                    }
                
                return result
                
        except Exception as e:
            print(f"❌ Email scan failed for {email}: {str(e)}")
            return {
                "name": None,
                "gender": None,
                "birthdate": None,
                "city": None,
                "success": False,
                "error": f"Email scan failed: {str(e)}"
            }

# ==================== MODULE EXPORTS ====================

__all__ = [
    'EmailService',
    'EmailScanResult',
    'is_lambda_environment'
]