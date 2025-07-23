import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Configuration class to manage environment variables"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    
    # Amadeus Configuration
    AMADEUS_API_KEY: str
    AMADEUS_API_SECRET: str
    AMADEUS_BASE_URL: str
    
    # Gmail API Configuration
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str
    
    # Outlook API Configuration
    OUTLOOK_CLIENT_ID: str
    OUTLOOK_CLIENT_SECRET: str
    OUTLOOK_REDIRECT_URI: str
    
    # AWS Configuration
    AWS_REGION: str
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        return cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            SUPABASE_URL=os.getenv("SUPABASE_URL", ""),
            SUPABASE_ANON_KEY=os.getenv("SUPABASE_ANON_KEY", ""),
            SUPABASE_SERVICE_ROLE_KEY=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
            AMADEUS_API_KEY=os.getenv("AMADEUS_API_KEY", ""),
            AMADEUS_API_SECRET=os.getenv("AMADEUS_API_SECRET", ""),
            AMADEUS_BASE_URL=os.getenv("AMADEUS_BASE_URL", "test"),
            GOOGLE_CLIENT_ID=os.getenv("GOOGLE_CLIENT_ID", ""),
            GOOGLE_CLIENT_SECRET=os.getenv("GOOGLE_CLIENT_SECRET", ""),
            GOOGLE_REDIRECT_URI=os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback"),
            OUTLOOK_CLIENT_ID=os.getenv("OUTLOOK_CLIENT_ID", ""),
            OUTLOOK_CLIENT_SECRET=os.getenv("OUTLOOK_CLIENT_SECRET", ""),
            OUTLOOK_REDIRECT_URI=os.getenv("OUTLOOK_REDIRECT_URI", "http://localhost:8000/auth/outlook/callback"),
            AWS_REGION=os.getenv("AWS_REGION", "us-east-1")
        )
    
    def validate(self, strict: bool = True) -> None:
        """
        Validate that required environment variables are set
        
        Args:
            strict: If True, raises errors for missing vars. If False, warns only.
        """
        required_fields = [
            "OPENAI_API_KEY",
            "SUPABASE_URL", 
            "SUPABASE_ANON_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "AMADEUS_API_KEY",
            "AMADEUS_API_SECRET"
        ]
        
        missing_fields = []
        for field in required_fields:
            if not getattr(self, field):
                missing_fields.append(field)
        
        if missing_fields:
            message = f"Environment variables not set: {', '.join(missing_fields)}"
            if strict:
                raise ValueError(message)
            else:
                print(f"Warning: {message} (using mock values for testing)")
    
    def is_production_ready(self) -> bool:
        """Check if configuration is ready for production use"""
        required_fields = [
            "OPENAI_API_KEY",
            "SUPABASE_URL", 
            "SUPABASE_ANON_KEY",
            "SUPABASE_SERVICE_ROLE_KEY"
        ]
        
        return all(getattr(self, field) for field in required_fields)

# Global configuration instance
config = Config.from_env()