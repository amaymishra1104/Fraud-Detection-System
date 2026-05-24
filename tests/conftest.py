"""
Pytest configuration and fixtures for fraud detection tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_transaction():
    """Sample transaction for testing."""
    return {
        "transaction_id": "TXN_00001",
        "amount": 150.00,
        "merchant": "Test Merchant",
        "location": "New York",
        "card_type": "Credit",
        "customer_age": 35,
        "transaction_type": "Online",
        "timestamp": "2024-05-17T10:30:00"
    }


@pytest.fixture
def sample_transactions_batch():
    """Sample batch of transactions for testing."""
    return [
        {
            "transaction_id": f"TXN_{i:05d}",
            "amount": 100 + (i * 10),
            "merchant": f"Merchant_{i}",
            "location": "New York",
            "card_type": "Credit",
            "customer_age": 25 + (i % 50),
            "transaction_type": "Online",
            "timestamp": "2024-05-17T10:30:00"
        }
        for i in range(10)
    ]
