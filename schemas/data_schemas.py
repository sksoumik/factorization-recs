"""
Data schemas module defining the data models.

This module provides Pydantic models for data validation and serialization:
1. User Features: Demographic and behavioral user attributes
2. Item Features: Product characteristics and metadata
3. Interaction Data: User-item interaction events and context
"""

from datetime import datetime

from pydantic import BaseModel, Field


class UserFeatures(BaseModel):
    """
    User features data model
    """

    user_id: str = Field(description="Unique identifier for the user")
    age: int = Field(
        ge=18, le=90, description="User's age in years, must be between 18 and 90"
    )
    gender: str = Field(description="User's gender identification")
    location: str = Field(description="User's geographical location or city")
    membership_level: str = Field(
        description="User's membership tier (e.g., Bronze, Silver, Gold)"
    )
    registration_date: datetime = Field(
        description="Timestamp when user registered on the platform"
    )
    device_type: str = Field(
        description="Primary device used (Desktop, Mobile, Tablet)"
    )
    browser: str = Field(description="Preferred web browser")
    is_mobile: bool = Field(
        description="Flag indicating if user primarily uses mobile devices"
    )
    avg_session_time: float = Field(description="Average session duration in minutes")
    total_purchases: int = Field(description="Total number of completed purchases")


class ItemFeatures(BaseModel):
    """
    Item features data model
    """

    item_id: str = Field(description="Unique identifier for the item")
    category: str = Field(description="Primary category of the item")
    subcategory: str = Field(description="Specific subcategory within main category")
    price: float = Field(description="Current price of the item")
    brand: str = Field(description="Brand or manufacturer name")
    avg_rating: float = Field(ge=0, le=5, description="Average user rating (0-5 scale)")
    total_reviews: int = Field(description="Total number of user reviews")
    in_stock: bool = Field(description="Current availability status")
    discount_percentage: float = Field(
        ge=0, le=100, description="Current discount percentage (0-100)"
    )
    weight: float = Field(description="Item weight in standard units")
    is_new: bool = Field(description="Flag indicating if item is new or used")


class Interaction(BaseModel):
    """
    User-item interaction data model
    """

    user_id: str = Field(description="Reference to the interacting user")
    item_id: str = Field(description="Reference to the interacted item")
    timestamp: datetime = Field(description="When the interaction occurred")
    interaction_type: str = Field(
        description="Type of interaction (view, cart, purchase)"
    )
    conversion: bool = Field(description="Whether interaction led to purchase")
    session_id: str = Field(description="Unique identifier for user session")
    interaction_duration: float = Field(
        description="Duration of interaction in seconds"
    )
    device_type: str = Field(description="Device used for interaction")
    price_at_interaction: float = Field(description="Item price at time of interaction")
