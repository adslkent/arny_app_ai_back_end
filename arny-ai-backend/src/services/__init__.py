"""
Services module for Arny AI Backend

This module provides external service integrations including:
- Amadeus API for flight and hotel search
- Email services for group invitations and profile scanning
- Third-party API wrappers and utilities

Usage:
    from services import AmadeusService, EmailService
    
    amadeus = AmadeusService()
    email = EmailService()
"""

from .amadeus_service import AmadeusService
from .email_service import EmailService

# Export main service classes
__all__ = [
    'AmadeusService',
    'EmailService'
]

# Version information
__version__ = '1.0.0'

# Service factory functions for easy instantiation
def create_amadeus_service():
    """
    Factory function to create an AmadeusService instance
    
    Returns:
        AmadeusService: Configured Amadeus service instance
    """
    return AmadeusService()

def create_email_service():
    """
    Factory function to create an EmailService instance
    
    Returns:
        EmailService: Configured email service instance
    """
    return EmailService()

# Service registry for dependency injection (if needed)
SERVICE_REGISTRY = {
    'amadeus': AmadeusService,
    'email': EmailService,
}

def get_service(service_name: str):
    """
    Get a service instance by name
    
    Args:
        service_name: Name of the service ('amadeus' or 'email')
        
    Returns:
        Service instance
        
    Raises:
        ValueError: If service name is not recognized
    """
    if service_name not in SERVICE_REGISTRY:
        raise ValueError(f"Unknown service: {service_name}. Available services: {list(SERVICE_REGISTRY.keys())}")
    
    service_class = SERVICE_REGISTRY[service_name]
    return service_class()

# Module-level constants
SUPPORTED_SERVICES = list(SERVICE_REGISTRY.keys())

# Convenience imports for backward compatibility
amadeus_service = AmadeusService
email_service = EmailService