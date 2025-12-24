from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.db.base import Base


class LastKnownLocation(Base):
    __tablename__ = "last_known_location"

    id = Column(Integer, primary_key=True, index=True)

    camera_id = Column(String, index=True)
    track_id = Column(Integer, index=True)
    cls = Column(String, index=True)        # "person", "backpack", etc.
    zone = Column(String, nullable=True)    # "Entrance", "Pantry"
    color = Column(String, nullable=True)   # for bags: "green", "black"

    x = Column(Float)  # bbox center or left, your choice
    y = Column(Float)
    w = Column(Float)
    h = Column(Float)

    last_seen_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
