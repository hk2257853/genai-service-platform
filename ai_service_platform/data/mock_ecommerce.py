"""Mock e-commerce databases, service registries, and promo configurations for agent tools."""

from typing import Dict, Any

ORDER_DATABASE: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "customer_name": "Alice Johnson",
        "status": "SHIPPED",
        "carrier": "FedEx",
        "tracking_number": "FDX-9988221100",
        "shipping_address": "742 Evergreen Terrace, Springfield, OR 97477",
        "estimated_delivery": "2026-08-25",
        "items": [
            {"product_id": "PROD-101", "name": "Sony WH-1000XM5 Wireless Headphones", "quantity": 1, "price": 399.99}
        ],
        "total_amount": 399.99
    },
    "ORD-2002": {
        "order_id": "ORD-2002",
        "customer_name": "Bob Smith",
        "status": "PROCESSING",
        "carrier": "UPS",
        "tracking_number": "UPS-5544332211",
        "shipping_address": "123 Main Street, Seattle, WA 98101",
        "estimated_delivery": "2026-08-28",
        "items": [
            {"product_id": "PROD-102", "name": "Logitech MX Master 3S Wireless Mouse", "quantity": 2, "price": 99.99}
        ],
        "total_amount": 199.98
    },
    "ORD-3003": {
        "order_id": "ORD-3003",
        "customer_name": "Carol Williams",
        "status": "DELIVERED",
        "carrier": "DHL Express",
        "tracking_number": "DHL-1122334455",
        "shipping_address": "456 Oak Avenue, Austin, TX 78701",
        "delivered_at": "2026-08-20T14:30:00Z",
        "estimated_delivery": "2026-08-20",
        "items": [
            {"product_id": "PROD-103", "name": "Keychron Q1 Pro Mechanical Keyboard", "quantity": 1, "price": 199.00},
            {"product_id": "PROD-104", "name": "Dell UltraSharp 27 4K Monitor", "quantity": 1, "price": 549.99}
        ],
        "total_amount": 748.99
    },
    "ORD-4004": {
        "order_id": "ORD-4004",
        "customer_name": "David Miller",
        "status": "CANCELLED",
        "carrier": "N/A",
        "tracking_number": "N/A",
        "shipping_address": "789 Pine Road, Boulder, CO 80302",
        "estimated_delivery": "N/A",
        "cancellation_reason": "Customer requested cancellation prior to dispatch.",
        "items": [
            {"product_id": "PROD-105", "name": "Apple MacBook Air M3", "quantity": 1, "price": 1099.00}
        ],
        "total_amount": 1099.00
    },
    "ORD-5005": {
        "order_id": "ORD-5005",
        "customer_name": "Emma Davis",
        "status": "OUT_FOR_DELIVERY",
        "carrier": "USPS",
        "tracking_number": "USPS-7788990011",
        "shipping_address": "321 Maple Blvd, Chicago, IL 60601",
        "estimated_delivery": "2026-08-23",
        "items": [
            {"product_id": "PROD-102", "name": "Logitech MX Master 3S Wireless Mouse", "quantity": 1, "price": 99.99}
        ],
        "total_amount": 99.99
    }
}

PRODUCT_DATABASE: Dict[str, Dict[str, Any]] = {
    "PROD-101": {
        "product_id": "PROD-101",
        "name": "Sony WH-1000XM5 Wireless Headphones",
        "category": "Audio & Electronics",
        "price": 399.99,
        "in_stock": True,
        "stock_quantity": 24,
        "specs": {
            "battery_life": "30 hours with ANC",
            "connectivity": "Bluetooth 5.2, 3.5mm jack",
            "weight": "250g",
            "color": "Black / Silver"
        },
        "return_policy_days": 30,
        "warranty": "1 year manufacturer limited warranty"
    },
    "PROD-102": {
        "product_id": "PROD-102",
        "name": "Logitech MX Master 3S Wireless Mouse",
        "category": "Computer Accessories",
        "price": 99.99,
        "in_stock": False,
        "stock_quantity": 0,
        "restock_eta": "2026-08-30",
        "specs": {
            "sensor": "8K DPI Quiet Click Sensor",
            "connectivity": "Bluetooth / Logi Bolt Receiver",
            "battery": "Up to 70 days rechargeable via USB-C"
        },
        "return_policy_days": 30,
        "warranty": "2 year hardware warranty"
    },
    "PROD-103": {
        "product_id": "PROD-103",
        "name": "Keychron Q1 Pro Mechanical Keyboard",
        "category": "Computer Accessories",
        "price": 199.00,
        "in_stock": True,
        "stock_quantity": 12,
        "specs": {
            "switches": "K Pro Red (Linear, Hot-swappable)",
            "connectivity": "Wireless Bluetooth 5.1 & Type-C Wired",
            "body": "Full CNC Aluminum Body, Gasket Mount"
        },
        "return_policy_days": 14,
        "warranty": "1 year warranty"
    },
    "PROD-104": {
        "product_id": "PROD-104",
        "name": "Dell UltraSharp 27 4K Monitor",
        "category": "Displays & Monitors",
        "price": 549.99,
        "in_stock": True,
        "stock_quantity": 7,
        "specs": {
            "resolution": "3840 x 2160 (4K UHD) @ 60Hz IPS",
            "ports": "USB-C 90W PD, HDMI 2.0, DP 1.4, USB 3.2 Hub",
            "color_gamut": "100% sRGB, 98% DCI-P3"
        },
        "return_policy_days": 30,
        "warranty": "3 year advanced exchange service"
    },
    "PROD-105": {
        "product_id": "PROD-105",
        "name": "Apple MacBook Air M3",
        "category": "Laptops & Computers",
        "price": 1099.00,
        "in_stock": True,
        "stock_quantity": 15,
        "specs": {
            "chip": "Apple M3 (8-core CPU, 10-core GPU)",
            "memory": "16GB Unified RAM",
            "storage": "512GB SSD",
            "display": "13.6-inch Liquid Retina Display"
        },
        "return_policy_days": 14,
        "warranty": "1 year AppleCare limited warranty"
    }
}

MICROSERVICE_REGISTRY: Dict[str, str] = {
    "payment_service": "HEALTHY (latency: 14ms, uptime: 99.98%)",
    "inventory_service": "HEALTHY (latency: 22ms, uptime: 99.95%)",
    "auth_service": "HEALTHY (latency: 8ms, uptime: 100%)",
    "notification_service": "DEGRADED (high queue delay: 350ms)"
}

PROMO_CODES: Dict[str, Dict[str, Any]] = {
    "VIP50": {"type": "percentage", "value": 0.50, "description": "50% off entire order"},
    "NEW20": {"type": "percentage", "value": 0.20, "description": "20% off for new customers"},
    "SAVE10": {"type": "fixed", "value": 10.00, "description": "$10 off order"}
}
