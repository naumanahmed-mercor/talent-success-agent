"""
MongoDB client for logging procedure selections and other analytics.
"""

import os
import logging
from typing import Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Configure logging
logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    Client for MongoDB operations.
    
    Provides methods to:
    - Log procedure selections
    - Store analytics data
    """
    
    def __init__(self, connection_uri: Optional[str] = None):
        """
        Initialize the MongoDB client.
        
        Args:
            connection_uri: MongoDB connection URI. If not provided, reads from MONGODB_URI env var.
        """
        self.connection_uri = connection_uri or os.getenv("MONGODB_URI")
        
        if not self.connection_uri:
            # No URI configured - create a dummy client that won't connect
            self.client = None
            self.db = None
            self.logs_collection = None
            logger.info("ℹ️  MongoDB URI not configured - logging disabled")
            return
        
        try:
            self.client = MongoClient(self.connection_uri)
            # Test connection
            self.client.admin.command('ping')
            logger.info("✅ MongoDB connection established successfully")
            
            # Database and collections
            self.db = self.client["procedures"]
            self.logs_collection = self.db["logs"]
        except PyMongoError as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.logs_collection = None
        
    def log_procedure_selection(
        self,
        procedure_id: str,
        conversation_id: str
    ) -> str:
        """
        Log a procedure selection to MongoDB.
        
        Args:
            procedure_id: ID of the selected procedure
            conversation_id: Intercom conversation ID
            
        Returns:
            String ID of the inserted document, or empty string if not connected
        """
        # Skip if not connected
        if self.logs_collection is None:
            return ""
        
        try:
            # Build document with only the two IDs and timestamp
            document = {
                "procedure_id": procedure_id,
                "conversation_id": conversation_id,
                "created_at": datetime.utcnow()
            }
            
            # Insert document
            result = self.logs_collection.insert_one(document)
            
            logger.info(
                f"✅ Logged procedure selection: "
                f"procedure_id={procedure_id}, conversation_id={conversation_id}"
            )
            
            return str(result.inserted_id)
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to log procedure selection: {e}")
            # Don't raise - logging should be non-critical
            return ""
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

