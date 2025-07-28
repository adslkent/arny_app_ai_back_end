"""
Database models for Arny AI

This module contains all Pydantic models used for data validation and serialization
in the Arny AI travel assistant application. These models correspond to the database
tables in Supabase (PostgreSQL) and provide type safety and validation.

Models included:
- UserProfile: User profile information
- OnboardingProgress: Onboarding state tracking
- GroupMember: Group/family membership
- ChatMessage: Conversation messages
- FlightSearch: Flight search results
- HotelSearch: Hotel search results
- UserPreferences: User travel preferences
- BookingRequest: Travel booking requests
- TravelItinerary: Complete travel plans

All models use Pydantic for validation and include proper type hints.
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date, timedelta
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from enum import Enum
import uuid
import re

# ==================== ENUMS ====================

class OnboardingStep(str, Enum):
    """Onboarding steps for tracking user progress"""
    GROUP_CODE = "group_code"
    EMAIL_SCAN = "email_scan"
    PERSONAL_INFO = "personal_info"
    JOB_DETAILS = "job_details"
    FINANCIAL_INFO = "financial_info"
    HOLIDAY_PREFERENCES = "holiday_preferences"
    GROUP_INVITES = "group_invites"
    COMPLETED = "completed"

class UserRole(str, Enum):
    """User roles in groups"""
    ADMIN = "admin"
    MEMBER = "member"

class MessageType(str, Enum):
    """Chat message types"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class CabinClass(str, Enum):
    """Flight cabin classes"""
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"

class BookingStatus(str, Enum):
    """Booking request status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class Gender(str, Enum):
    """Gender options"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

# ==================== BASE MODELS ====================

class TimestampMixin(BaseModel):
    """Mixin for models with timestamp fields"""
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

class BaseModelWithId(BaseModel):
    """Base model with UUID ID field"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# ==================== USER MODELS ====================

class UserProfile(TimestampMixin):
    """
    User profile data model
    
    Contains all user information collected during onboarding and usage.
    Maps to the 'user_profiles' table in Supabase.
    """
    user_id: str = Field(..., description="Supabase user UUID")
    email: EmailStr = Field(..., description="User's email address")
    
    # Personal Information
    name: Optional[str] = Field(None, max_length=255, description="Full name")
    gender: Optional[Gender] = Field(None, description="Gender")
    birthdate: Optional[date] = Field(None, description="Date of birth")
    city: Optional[str] = Field(None, max_length=255, description="Current city")
    
    # Work Information
    employer: Optional[str] = Field(None, max_length=255, description="Employer name")
    working_schedule: Optional[str] = Field(None, description="Working schedule details")
    holiday_frequency: Optional[str] = Field(None, description="How often they take holidays")
    
    # Financial Information
    annual_income: Optional[str] = Field(None, description="Annual income range")
    monthly_spending: Optional[str] = Field(None, description="Average monthly spending")
    
    # Travel Preferences
    holiday_preferences: Optional[List[str]] = Field(default_factory=list, description="Preferred holiday activities")
    
    # Group Information
    group_code: Optional[str] = Field(None, max_length=10, description="Group code for family/group travel")
    
    # Status
    onboarding_completed: bool = Field(default=False, description="Whether onboarding is complete")
    is_active: bool = Field(default=True, description="Whether account is active")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate name format"""
        if v and len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters')
        return v.strip() if v else v
    
    @field_validator('birthdate')
    @classmethod
    def validate_birthdate(cls, v):
        """Validate birthdate is not in the future and user is not too young"""
        if v:
            today = date.today()
            if v > today:
                raise ValueError('Birthdate cannot be in the future')
            
            age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
            if age < 13:
                raise ValueError('User must be at least 13 years old')
            if age > 120:
                raise ValueError('Invalid birthdate')
        return v
    
    @field_validator('group_code')
    @classmethod
    def validate_group_code(cls, v):
        """Validate group code format"""
        if v:
            if not re.match(r'^[A-Z0-9]{4,10}$', v):
                raise ValueError('Group code must be 4-10 alphanumeric characters')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

class OnboardingProgress(TimestampMixin):
    """
    Onboarding progress tracking model
    
    Tracks user progress through the onboarding flow.
    Maps to the 'onboarding_progress' table in Supabase.
    """
    user_id: str = Field(..., description="Supabase user UUID")
    current_step: OnboardingStep = Field(default=OnboardingStep.GROUP_CODE, description="Current onboarding step")
    collected_data: Dict[str, Any] = Field(default_factory=dict, description="Data collected so far")
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @model_validator(mode='before')
    @classmethod
    def calculate_completion_percentage(cls, values):
        """Calculate completion percentage based on current step"""
        if isinstance(values, dict):
            step_percentages = {
                OnboardingStep.GROUP_CODE: 10.0,
                OnboardingStep.EMAIL_SCAN: 20.0,
                OnboardingStep.PERSONAL_INFO: 40.0,
                OnboardingStep.JOB_DETAILS: 60.0,
                OnboardingStep.FINANCIAL_INFO: 80.0,
                OnboardingStep.HOLIDAY_PREFERENCES: 90.0,
                OnboardingStep.GROUP_INVITES: 95.0,
                OnboardingStep.COMPLETED: 100.0
            }
            
            current_step = values.get('current_step', OnboardingStep.GROUP_CODE)
            values['completion_percentage'] = step_percentages.get(current_step, 0.0)
        return values

class UserPreferences(TimestampMixin):
    """
    User travel preferences model
    
    Stores detailed travel preferences for personalization.
    Maps to the 'user_preferences' table in Supabase.
    """
    user_id: str = Field(..., description="Supabase user UUID")
    
    # Travel Preferences
    preferred_airlines: Optional[List[str]] = Field(default_factory=list, description="Preferred airline codes")
    preferred_hotels: Optional[List[str]] = Field(default_factory=list, description="Preferred hotel chains")
    preferred_cabin_class: Optional[CabinClass] = Field(default=CabinClass.ECONOMY, description="Preferred cabin class")
    
    # Budget Preferences
    budget_range: Optional[Dict[str, float]] = Field(default_factory=dict, description="Budget ranges for different trip types")
    price_sensitivity: Optional[str] = Field(None, description="How price-sensitive the user is")
    
    # Dietary and Accessibility
    dietary_restrictions: Optional[List[str]] = Field(default_factory=list, description="Dietary restrictions")
    accessibility_needs: Optional[List[str]] = Field(default_factory=list, description="Accessibility requirements")
    
    # Travel Style
    trip_types: Optional[List[str]] = Field(default_factory=list, description="Types of trips they prefer")
    
    # Notification Preferences
    email_notifications: bool = Field(default=True, description="Receive email notifications")
    push_notifications: bool = Field(default=True, description="Receive push notifications")
    deal_alerts: bool = Field(default=True, description="Receive deal alerts")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @field_validator('budget_range')
    @classmethod
    def validate_budget_range(cls, v):
        """Validate budget range values"""
        if v:
            for key, value in v.items():
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f'Budget value for {key} must be a positive number')
        return v

# ==================== GROUP MODELS ====================

class GroupMember(BaseModelWithId, TimestampMixin):
    """
    Group member model for family/group functionality
    
    Tracks group membership and roles.
    Maps to the 'group_members' table in Supabase.
    """
    group_code: str = Field(..., max_length=10, description="Group code")
    user_id: str = Field(..., description="Supabase user UUID")
    role: UserRole = Field(default=UserRole.MEMBER, description="User role in group")
    joined_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="When user joined group")
    is_active: bool = Field(default=True, description="Whether membership is active")
    
    @field_validator('group_code')
    @classmethod
    def validate_group_code(cls, v):
        """Validate group code format"""
        if not re.match(r'^[A-Z0-9]{4,10}$', v):
            raise ValueError('Group code must be 4-10 alphanumeric characters')
        return v
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== CHAT MODELS ====================

class ChatMessage(BaseModelWithId):
    """
    Chat message model for storing conversations
    
    Stores all chat interactions between users and the AI.
    Maps to the 'chat_messages' table in Supabase.
    """
    user_id: str = Field(..., description="Supabase user UUID")
    session_id: str = Field(..., description="Chat session UUID")
    message_type: MessageType = Field(..., description="Type of message")
    content: str = Field(..., min_length=1, description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional message metadata")
    agent_type: Optional[str] = Field(None, description="Which agent handled this message")
    response_time_ms: Optional[int] = Field(None, ge=0, description="Response time in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        """Validate session_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('session_id must be a valid UUID')
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate message content"""
        if len(v.strip()) == 0:
            raise ValueError('Message content cannot be empty')
        if len(v) > 10000:
            raise ValueError('Message content too long (max 10000 characters)')
        return v.strip()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== SEARCH MODELS ====================

class FlightSearch(BaseModel):  # FIXED: Remove BaseModelWithId inheritance
    """
    Flight search result model
    
    Stores flight search parameters and results.
    Maps to the 'flight_searches' table in Supabase.
    """
    search_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Flight search UUID")  # FIXED: Added search_id field
    user_id: str = Field(..., description="Supabase user UUID")
    session_id: Optional[str] = Field(None, description="Chat session UUID")
    
    # Search Parameters
    origin: str = Field(..., min_length=3, max_length=3, description="Origin airport code")
    destination: str = Field(..., min_length=3, max_length=3, description="Destination airport code")
    departure_date: date = Field(..., description="Departure date")
    return_date: Optional[date] = Field(None, description="Return date for round-trip")
    passengers: int = Field(default=1, ge=1, le=9, description="Number of passengers")
    cabin_class: CabinClass = Field(default=CabinClass.ECONOMY, description="Cabin class")
    
    # Search Results
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Flight search results from Amadeus")
    result_count: int = Field(default=0, ge=0, description="Number of results found")
    search_duration_ms: Optional[int] = Field(None, ge=0, description="Search duration in milliseconds")
    
    # Status
    search_successful: bool = Field(default=False, description="Whether search was successful")
    error_message: Optional[str] = Field(None, description="Error message if search failed")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @field_validator('origin', 'destination')
    @classmethod
    def validate_airport_code(cls, v):
        """Validate airport code format"""
        if not re.match(r'^[A-Z]{3}$', v.upper()):
            raise ValueError('Airport code must be 3 uppercase letters')
        return v.upper()
    
    @field_validator('departure_date')
    @classmethod
    def validate_departure_date(cls, v):
        """Validate departure date is not in the past"""
        if v < date.today():
            raise ValueError('Departure date cannot be in the past')
        return v
    
    @field_validator('return_date')
    @classmethod
    def validate_return_date(cls, v, info):
        """Validate return date is after departure date"""
        if v and 'departure_date' in info.data:
            if v <= info.data['departure_date']:
                raise ValueError('Return date must be after departure date')
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_search_results(cls, values):
        """Update result count based on search results"""
        if isinstance(values, dict):
            search_results = values.get('search_results', [])
            values['result_count'] = len(search_results)
            values['search_successful'] = len(search_results) > 0 and not values.get('error_message')
        return values
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

class HotelSearch(BaseModelWithId):
    """
    Hotel search result model
    
    Stores hotel search parameters and results.
    Maps to the 'hotel_searches' table in Supabase.
    """
    search_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Hotel search UUID")
    user_id: str = Field(..., description="Supabase user UUID")
    session_id: Optional[str] = Field(None, description="Chat session UUID")
    
    # Search Parameters
    city_code: str = Field(..., min_length=3, max_length=3, description="City code")
    check_in_date: date = Field(..., description="Check-in date")
    check_out_date: date = Field(..., description="Check-out date")
    adults: int = Field(default=1, ge=1, le=30, description="Number of adults")
    rooms: int = Field(default=1, ge=1, le=10, description="Number of rooms")
    
    # Search Results
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Hotel search results from Amadeus")
    result_count: int = Field(default=0, ge=0, description="Number of results found")
    search_duration_ms: Optional[int] = Field(None, ge=0, description="Search duration in milliseconds")
    
    # Status
    search_successful: bool = Field(default=False, description="Whether search was successful")
    error_message: Optional[str] = Field(None, description="Error message if search failed")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @field_validator('city_code')
    @classmethod
    def validate_city_code(cls, v):
        """Validate city code format"""
        if not re.match(r'^[A-Z]{3}$', v.upper()):
            raise ValueError('City code must be 3 uppercase letters')
        return v.upper()
    
    @field_validator('check_in_date')
    @classmethod
    def validate_check_in_date(cls, v):
        """Validate check-in date is not in the past"""
        if v < date.today():
            raise ValueError('Check-in date cannot be in the past')
        return v
    
    @field_validator('check_out_date')
    @classmethod
    def validate_check_out_date(cls, v, info):
        """Validate check-out date is after check-in date"""
        if 'check_in_date' in info.data:
            if v <= info.data['check_in_date']:
                raise ValueError('Check-out date must be after check-in date')
        return v
    
    @model_validator(mode='before')
    @classmethod
    def validate_search_results(cls, values):
        """Update result count based on search results"""
        if isinstance(values, dict):
            search_results = values.get('search_results', [])
            values['result_count'] = len(search_results)
            values['search_successful'] = len(search_results) > 0 and not values.get('error_message')
        return values
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

# ==================== BOOKING MODELS ====================

class BookingRequest(BaseModelWithId, TimestampMixin):
    """
    Travel booking request model
    
    Stores booking requests and their status.
    """
    user_id: str = Field(..., description="Supabase user UUID")
    session_id: Optional[str] = Field(None, description="Chat session UUID")
    
    # Booking Details
    booking_type: str = Field(..., description="Type of booking (flight, hotel, package)")
    provider: str = Field(..., description="Service provider (amadeus, etc.)")
    external_id: Optional[str] = Field(None, description="External booking reference")
    
    # Booking Data
    booking_data: Dict[str, Any] = Field(..., description="Complete booking information")
    total_amount: Optional[float] = Field(None, ge=0, description="Total booking amount")
    currency: Optional[str] = Field(None, max_length=3, description="Currency code")
    
    # Status
    status: BookingStatus = Field(default=BookingStatus.PENDING, description="Booking status")
    confirmation_code: Optional[str] = Field(None, description="Booking confirmation code")
    payment_status: Optional[str] = Field(None, description="Payment status")
    
    # Timestamps
    booking_date: Optional[datetime] = Field(None, description="When booking was made")
    travel_date: Optional[date] = Field(None, description="Travel start date")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v):
        """Validate currency code format"""
        if v and not re.match(r'^[A-Z]{3}$', v):
            raise ValueError('Currency must be 3 uppercase letters')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

class TravelItinerary(BaseModelWithId, TimestampMixin):
    """
    Complete travel itinerary model
    
    Stores complete travel plans including flights, hotels, and activities.
    """
    user_id: str = Field(..., description="Supabase user UUID")
    group_code: Optional[str] = Field(None, description="Associated group code")
    
    # Itinerary Details
    title: str = Field(..., min_length=1, max_length=255, description="Itinerary title")
    description: Optional[str] = Field(None, description="Itinerary description")
    destination: str = Field(..., description="Main destination")
    
    # Dates
    start_date: date = Field(..., description="Trip start date")
    end_date: date = Field(..., description="Trip end date")
    duration_days: int = Field(..., ge=1, description="Trip duration in days")
    
    # Components
    flights: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Flight bookings")
    hotels: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Hotel bookings")
    activities: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Activities and tours")
    
    # Budget
    estimated_cost: Optional[float] = Field(None, ge=0, description="Estimated total cost")
    actual_cost: Optional[float] = Field(None, ge=0, description="Actual total cost")
    currency: Optional[str] = Field(None, max_length=3, description="Currency code")
    
    # Status
    is_booked: bool = Field(default=False, description="Whether all components are booked")
    is_shared: bool = Field(default=False, description="Whether itinerary is shared with group")
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a valid UUID"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')
    
    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, info):
        """Validate end date is after start date"""
        if 'start_date' in info.data:
            if v <= info.data['start_date']:
                raise ValueError('End date must be after start date')
        return v
    
    @model_validator(mode='before')
    @classmethod
    def calculate_duration(cls, values):
        """Calculate duration in days"""
        if isinstance(values, dict):
            start_date = values.get('start_date')
            end_date = values.get('end_date')
            
            if start_date and end_date:
                duration = (end_date - start_date).days
                values['duration_days'] = max(1, duration)
        
        return values
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

# ==================== UTILITY MODELS ====================

class APIResponse(BaseModel):
    """
    Standard API response model
    
    Used for consistent API responses across the application.
    """
    success: bool = Field(..., description="Whether the operation was successful")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PaginationInfo(BaseModel):
    """
    Pagination information model
    
    Used for paginated API responses.
    """
    page: int = Field(default=1, ge=1, description="Current page number")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")
    total_items: int = Field(default=0, ge=0, description="Total number of items")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")
    has_next: bool = Field(default=False, description="Whether there is a next page")
    has_previous: bool = Field(default=False, description="Whether there is a previous page")
    
    @model_validator(mode='before')
    @classmethod
    def calculate_pagination(cls, values):
        """Calculate pagination fields"""
        if isinstance(values, dict):
            total_items = values.get('total_items', 0)
            per_page = values.get('per_page', 20)
            page = values.get('page', 1)
            
            total_pages = (total_items + per_page - 1) // per_page if total_items > 0 else 0
            has_next = page < total_pages
            has_previous = page > 1
            
            values['total_pages'] = total_pages
            values['has_next'] = has_next
            values['has_previous'] = has_previous
        
        return values

# ==================== MODEL EXPORTS ====================

__all__ = [
    # Enums
    'OnboardingStep',
    'UserRole', 
    'MessageType',
    'CabinClass',
    'BookingStatus',
    'Gender',
    
    # Base Models
    'TimestampMixin',
    'BaseModelWithId',
    
    # User Models
    'UserProfile',
    'OnboardingProgress',
    'UserPreferences',
    
    # Group Models
    'GroupMember',
    
    # Chat Models
    'ChatMessage',
    
    # Search Models
    'FlightSearch',
    'HotelSearch',
    
    # Booking Models
    'BookingRequest',
    'TravelItinerary',
    
    # Utility Models
    'APIResponse',
    'PaginationInfo'
]
