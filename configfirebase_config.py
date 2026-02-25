"""
Firebase configuration and initialization with proper error handling.
Critical: Centralized Firebase management for the entire AATN ecosystem.
"""
import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.firestore_v1.client import Client as FirestoreClient
from google.cloud.firestore_v1.base_client import BaseClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FirebaseConfig:
    """Configuration dataclass for Firebase initialization."""
    project_id: str
    credentials_path: Optional[str] = None
    database_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'FirebaseConfig':
        """Initialize config from environment variables with validation."""
        try:
            project_id = os.getenv('FIREBASE_PROJECT_ID')
            if not project_id:
                raise ValueError("FIREBASE_PROJECT_ID environment variable not set")
                
            return cls(
                project_id=project_id,
                credentials_path=os.getenv('FIREBASE_CREDENTIALS_PATH'),
                database_url=os.getenv('FIREBASE_DATABASE_URL')
            )
        except Exception as e:
            logger.error(f"Failed to load Firebase config from env: {e}")
            raise

class FirebaseManager:
    """Singleton manager for Firebase connections with self-healing capabilities."""
    
    _instance: Optional['FirebaseManager'] = None
    _firestore_client: Optional[FirestoreClient] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = FirebaseConfig.from_env()
            self._initialize_firebase()
            self._initialized = True
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase with proper error handling and fallbacks."""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            # Check if Firebase app already exists
            if not firebase_admin._apps:
                if self.config.credentials_path and os.path.exists(self.config.credentials_path):
                    cred = credentials.Certificate(self.config.credentials_path)
                    logger.info(f"Initializing Firebase with credentials from {self.config.credentials_path}")
                else:
                    # Use default application credentials
                    cred = credentials.ApplicationDefault()
                    logger.info("Initializing Firebase with default application credentials")
                
                # Initialize app
                app_options = {'projectId': self.config.project_id}
                if self.config.database_url:
                    app_options['databaseURL'] = self.config.database_url
                
                firebase_admin.initialize_app(cred, app_options)
            
            # Initialize Firestore client
            self._firestore_client = firestore.client()
            logger.info(f"Firebase initialized successfully for project: {self.config.project_id}")
            
        except (DefaultCredentialsError, FileNotFoundError) as e:
            logger.error(f"Firebase credentials error: {e}")
            raise ConnectionError(f"Failed to initialize Firebase: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing Firebase: {e}")
            raise
    
    @property
    def firestore(self) -> FirestoreClient:
        """Get Firestore client with connection validation."""
        if self._firestore_client is None:
            logger.warning("Firestore client not initialized, attempting reinitialization")
            self._initialize_firebase()
            
            if self._firestore_client is None:
                raise ConnectionError("Firestore client unavailable after reinitialization")
        
        return self._firestore_client
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Firebase connection."""
        try:
            # Test Firestore connection with a simple operation
            collections = list(self.firestore.collections())
            return {
                'status': 'healthy',
                'project_id': self.config.project_id,
                'collections_count': len(collections),
                'timestamp': firestore.SERVER_TIMESTAMP
            }
        except Exception as e:
            logger.error(f"Firebase health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': firestore.SERVER_TIMESTAMP
            }
    
    def cleanup(self) -> None:
        """Cleanup Firebase connections."""
        try:
            if self._firestore_client:
                self._firestore_client.close()
                self._firestore_client = None
                logger.info("Firebase connections cleaned up")
        except Exception as e:
            logger.error(f"Error during Firebase cleanup: {e}")