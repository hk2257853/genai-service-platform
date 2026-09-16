"""Standalone FastMCP Server exposing E-Commerce Order and Product tools over STDIO."""

import sys
import json
import logging
from mcp.server.fastmcp import FastMCP
from ai_service_platform.data.mock_ecommerce import ORDER_DATABASE, PRODUCT_DATABASE

# Ensure all logging goes strictly to stderr to prevent corrupting stdout (JSON-RPC pipe)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ecommerce_mcp_server")

mcp = FastMCP("EcommerceService")


@mcp.tool()
def get_order_details(order_id: str) -> str:
    """Retrieve fulfillment status, carrier, tracking number, items, and total cost for an order by ID (e.g. ORD-1001)."""
    norm_id = str(order_id).strip().upper()
    logger.info("Handling get_order_details for ID: %s", norm_id)
    
    if norm_id not in ORDER_DATABASE:
        return json.dumps({
            "error": f"Order '{order_id}' was not found in the database.",
            "available_order_ids": list(ORDER_DATABASE.keys())
        })
        
    return json.dumps(ORDER_DATABASE[norm_id])


@mcp.tool()
def get_product_details(product_id_or_name: str) -> str:
    """Retrieve product specs, price, stock inventory availability, warranty, and return policy by SKU (e.g. PROD-101) or product name."""
    query = str(product_id_or_name).strip()
    query_upper = query.upper()
    logger.info("Handling get_product_details for query: %s", query)
    
    # 1. Exact SKU match
    if query_upper in PRODUCT_DATABASE:
        return json.dumps(PRODUCT_DATABASE[query_upper])
        
    # 2. Case-insensitive substring match
    query_lower = query.lower()
    for prod in PRODUCT_DATABASE.values():
        if query_lower in prod["name"].lower() or prod["name"].lower() in query_lower:
            return json.dumps(prod)
            
    # 3. Token overlap match
    query_tokens = set(query_lower.split())
    for prod in PRODUCT_DATABASE.values():
        prod_tokens = set(prod["name"].lower().split())
        if len(query_tokens.intersection(prod_tokens)) >= 2:
            return json.dumps(prod)

    return json.dumps({
        "error": f"Product '{product_id_or_name}' not found.",
        "available_skus": list(PRODUCT_DATABASE.keys())
    })


if __name__ == "__main__":
    logger.info("Starting Ecommerce FastMCP Server over STDIO...")
    mcp.run(transport="stdio")
