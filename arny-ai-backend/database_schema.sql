-- Arny AI Database Schema - Clean Recreation
-- This will drop existing tables and recreate with complete schema
-- ⚠️ WARNING: This will delete all existing data!

-- Drop existing tables in correct order (reverse dependency order)
DROP TABLE IF EXISTS travel_itineraries CASCADE;
DROP TABLE IF EXISTS booking_requests CASCADE;
DROP TABLE IF EXISTS user_preferences CASCADE;
DROP TABLE IF EXISTS hotel_searches CASCADE;
DROP TABLE IF EXISTS flight_searches CASCADE;
DROP TABLE IF EXISTS chat_messages CASCADE;
DROP TABLE IF EXISTS group_members CASCADE;
DROP TABLE IF EXISTS onboarding_progress CASCADE;
DROP TABLE IF EXISTS user_profiles CASCADE;

-- Drop existing functions
DROP FUNCTION IF EXISTS get_user_analytics(uuid, integer);
DROP FUNCTION IF EXISTS group_code_exists(text);
DROP FUNCTION IF EXISTS get_group_members(text);
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User Profiles Table
CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    gender VARCHAR(50),
    birthdate DATE,
    city VARCHAR(255),
    employer VARCHAR(255),
    working_schedule TEXT,
    holiday_frequency VARCHAR(255),
    annual_income VARCHAR(255),
    monthly_spending VARCHAR(255),
    holiday_preferences JSONB DEFAULT '[]'::jsonb,
    travel_style VARCHAR(50), -- 'budget', 'comfort', 'luxury'
    group_code VARCHAR(10),
    onboarding_completed BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Onboarding Progress Table
CREATE TABLE onboarding_progress (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    current_step VARCHAR(50) NOT NULL DEFAULT 'group_code',
    collected_data JSONB DEFAULT '{}'::jsonb,
    completion_percentage REAL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Group Members Table
CREATE TABLE group_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    group_code VARCHAR(10) NOT NULL,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'member', -- 'admin' or 'member'
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(group_code, user_id)
);

-- Chat Messages Table
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    message_type VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    agent_type VARCHAR(50),
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Flight Searches Table
CREATE TABLE flight_searches (
    search_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID,
    origin VARCHAR(10) NOT NULL,
    destination VARCHAR(10) NOT NULL,
    departure_date DATE NOT NULL,
    return_date DATE,
    passengers INTEGER DEFAULT 1,
    cabin_class VARCHAR(20) DEFAULT 'ECONOMY',
    search_results JSONB DEFAULT '[]'::jsonb,
    result_count INTEGER DEFAULT 0,
    search_duration_ms INTEGER,
    search_successful BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Hotel Searches Table
CREATE TABLE hotel_searches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID,
    city_code VARCHAR(10) NOT NULL,
    check_in_date DATE NOT NULL,
    check_out_date DATE NOT NULL,
    adults INTEGER DEFAULT 1,
    rooms INTEGER DEFAULT 1,
    search_results JSONB DEFAULT '[]'::jsonb,
    result_count INTEGER DEFAULT 0,
    search_duration_ms INTEGER,
    search_successful BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User Preferences Table (for advanced personalization)
CREATE TABLE user_preferences (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    preferred_airlines JSONB DEFAULT '[]'::jsonb,
    preferred_hotels JSONB DEFAULT '[]'::jsonb,
    preferred_cabin_class VARCHAR(20) DEFAULT 'ECONOMY',
    budget_range JSONB DEFAULT '{}'::jsonb,
    price_sensitivity VARCHAR(50),
    dietary_restrictions JSONB DEFAULT '[]'::jsonb,
    accessibility_needs JSONB DEFAULT '[]'::jsonb,
    travel_style VARCHAR(50), -- 'budget', 'comfort', 'luxury'
    trip_types JSONB DEFAULT '[]'::jsonb,
    email_notifications BOOLEAN DEFAULT TRUE,
    push_notifications BOOLEAN DEFAULT TRUE,
    deal_alerts BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Booking Requests Table
CREATE TABLE booking_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID,
    booking_type VARCHAR(50) NOT NULL, -- 'flight', 'hotel', 'package'
    provider VARCHAR(100) NOT NULL, -- 'amadeus', etc.
    external_id VARCHAR(255),
    booking_data JSONB NOT NULL,
    total_amount DECIMAL(10,2),
    currency VARCHAR(3),
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'confirmed', 'cancelled', 'failed'
    confirmation_code VARCHAR(100),
    payment_status VARCHAR(50),
    booking_date TIMESTAMP WITH TIME ZONE,
    travel_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Travel Itineraries Table
CREATE TABLE travel_itineraries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    group_code VARCHAR(10),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    destination VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    duration_days INTEGER NOT NULL,
    flights JSONB DEFAULT '[]'::jsonb,
    hotels JSONB DEFAULT '[]'::jsonb,
    activities JSONB DEFAULT '[]'::jsonb,
    estimated_cost DECIMAL(10,2),
    actual_cost DECIMAL(10,2),
    currency VARCHAR(3),
    is_booked BOOLEAN DEFAULT FALSE,
    is_shared BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_user_profiles_email ON user_profiles(email);
CREATE INDEX idx_user_profiles_group_code ON user_profiles(group_code);
CREATE INDEX idx_user_profiles_is_active ON user_profiles(is_active);
CREATE INDEX idx_group_members_group_code ON group_members(group_code);
CREATE INDEX idx_group_members_user_id ON group_members(user_id);
CREATE INDEX idx_group_members_is_active ON group_members(is_active);
CREATE INDEX idx_chat_messages_user_session ON chat_messages(user_id, session_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);
CREATE INDEX idx_chat_messages_agent_type ON chat_messages(agent_type);
CREATE INDEX idx_flight_searches_user_id ON flight_searches(user_id);
CREATE INDEX idx_flight_searches_created_at ON flight_searches(created_at);
CREATE INDEX idx_flight_searches_session_id ON flight_searches(session_id);
CREATE INDEX idx_hotel_searches_user_id ON hotel_searches(user_id);
CREATE INDEX idx_hotel_searches_created_at ON hotel_searches(created_at);
CREATE INDEX idx_hotel_searches_session_id ON hotel_searches(session_id);
CREATE INDEX idx_booking_requests_user_id ON booking_requests(user_id);
CREATE INDEX idx_booking_requests_status ON booking_requests(status);
CREATE INDEX idx_booking_requests_booking_type ON booking_requests(booking_type);
CREATE INDEX idx_travel_itineraries_user_id ON travel_itineraries(user_id);
CREATE INDEX idx_travel_itineraries_group_code ON travel_itineraries(group_code);
CREATE INDEX idx_travel_itineraries_start_date ON travel_itineraries(start_date);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_user_profiles_updated_at 
    BEFORE UPDATE ON user_profiles 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_onboarding_progress_updated_at 
    BEFORE UPDATE ON onboarding_progress 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_group_members_updated_at 
    BEFORE UPDATE ON group_members 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at 
    BEFORE UPDATE ON user_preferences 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_booking_requests_updated_at 
    BEFORE UPDATE ON booking_requests 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_travel_itineraries_updated_at 
    BEFORE UPDATE ON travel_itineraries 
    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

-- Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE onboarding_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE group_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE flight_searches ENABLE ROW LEVEL SECURITY;
ALTER TABLE hotel_searches ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE booking_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE travel_itineraries ENABLE ROW LEVEL SECURITY;

-- RLS Policies for user_profiles
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own profile" ON user_profiles
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for onboarding_progress
CREATE POLICY "Users can view own onboarding progress" ON onboarding_progress
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own onboarding progress" ON onboarding_progress
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own onboarding progress" ON onboarding_progress
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for group_members
CREATE POLICY "Users can view own group membership" ON group_members
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can view group members of same group" ON group_members
    FOR SELECT USING (
        group_code IN (
            SELECT gm.group_code 
            FROM group_members gm 
            WHERE gm.user_id = auth.uid() AND gm.is_active = true
        )
    );

CREATE POLICY "Users can insert own group membership" ON group_members
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own group membership" ON group_members
    FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for chat_messages
CREATE POLICY "Users can view own chat messages" ON chat_messages
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own chat messages" ON chat_messages
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for flight_searches
CREATE POLICY "Users can view own flight searches" ON flight_searches
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own flight searches" ON flight_searches
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for hotel_searches
CREATE POLICY "Users can view own hotel searches" ON hotel_searches
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own hotel searches" ON hotel_searches
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for user_preferences
CREATE POLICY "Users can view own preferences" ON user_preferences
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can update own preferences" ON user_preferences
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own preferences" ON user_preferences
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- RLS Policies for booking_requests
CREATE POLICY "Users can view own booking requests" ON booking_requests
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own booking requests" ON booking_requests
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own booking requests" ON booking_requests
    FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for travel_itineraries
CREATE POLICY "Users can view own itineraries" ON travel_itineraries
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can view shared group itineraries" ON travel_itineraries
    FOR SELECT USING (
        is_shared = true AND group_code IN (
            SELECT gm.group_code 
            FROM group_members gm 
            WHERE gm.user_id = auth.uid() AND gm.is_active = true
        )
    );

CREATE POLICY "Users can insert own itineraries" ON travel_itineraries
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own itineraries" ON travel_itineraries
    FOR UPDATE USING (auth.uid() = user_id);

-- Create a function to get group members (callable from backend)
CREATE OR REPLACE FUNCTION get_group_members(target_group_code text)
RETURNS TABLE (
    user_id uuid,
    role text,
    joined_at timestamptz
) 
LANGUAGE plpgsql
SECURITY DEFINER -- Run with elevated privileges
AS $$
BEGIN
    RETURN QUERY
    SELECT gm.user_id, gm.role, gm.joined_at
    FROM group_members gm
    WHERE gm.group_code = target_group_code AND gm.is_active = true;
END;
$$;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_group_members(text) TO authenticated;

-- Create a function to check if group code exists
CREATE OR REPLACE FUNCTION group_code_exists(target_group_code text)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 
        FROM group_members 
        WHERE group_code = target_group_code AND is_active = true
    );
END;
$$;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION group_code_exists(text) TO authenticated;

-- Create a function to get user analytics
CREATE OR REPLACE FUNCTION get_user_analytics(target_user_id uuid, days_back integer DEFAULT 30)
RETURNS TABLE (
    total_messages integer,
    flight_searches integer,
    hotel_searches integer,
    booking_requests integer
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    start_date timestamptz;
BEGIN
    start_date := NOW() - (days_back || ' days')::interval;
    
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*)::integer FROM chat_messages WHERE user_id = target_user_id AND created_at >= start_date),
        (SELECT COUNT(*)::integer FROM flight_searches WHERE user_id = target_user_id AND created_at >= start_date),
        (SELECT COUNT(*)::integer FROM hotel_searches WHERE user_id = target_user_id AND created_at >= start_date),
        (SELECT COUNT(*)::integer FROM booking_requests WHERE user_id = target_user_id AND created_at >= start_date);
END;
$$;

-- Grant execute permission on the analytics function
GRANT EXECUTE ON FUNCTION get_user_analytics(uuid, integer) TO authenticated;