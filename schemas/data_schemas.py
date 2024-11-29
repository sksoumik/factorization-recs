from datetime import datetime

from pydantic import BaseModel, Field


class UserFeatures(BaseModel):
    """User features data model."""

    user_id: str
    age: int = Field(ge=18, le=90)
    gender: str
    location: str
    membership_level: str
    registration_date: datetime
    device_type: str
    browser: str
    is_mobile: bool
    avg_session_time: float
    total_purchases: int


class ItemFeatures(BaseModel):
    """Item features data model."""

    item_id: str
    category: str
    subcategory: str
    price: float
    brand: str
    avg_rating: float = Field(ge=0, le=5)
    total_reviews: int
    in_stock: bool
    discount_percentage: float = Field(ge=0, le=100)
    weight: float
    is_new: bool


class Interaction(BaseModel):
    """User-item interaction data model."""

    user_id: str
    item_id: str
    timestamp: datetime
    interaction_type: str  # view, cart, purchase
    conversion: bool
    session_id: str
    interaction_duration: float
    device_type: str
    price_at_interaction: float
