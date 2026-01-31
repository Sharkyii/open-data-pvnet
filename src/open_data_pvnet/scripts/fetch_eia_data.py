import os
import logging
import requests
import pandas as pd
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class EIAData:
    """
    Class to handle interactions with the EIA API v2.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            logger.warning("EIA_API_KEY environment variable is not set. You must provide an API key to fetch data.")
        self.base_url = "https://api.eia.gov/v2"

    def get_data(
        self, 
        route: str, 
        frequency: str, 
        start_date: str, 
        end_date: str, 
        data_cols: List[str] = ["value"],
        facets: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        length: int = 5000
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from the EIA API.
        
        Args:
            route: API route (e.g. 'electricity/rto/daily-fuel-type-data')
            frequency: Data frequency (e.g. 'daily', 'hourly')
            start_date: Start date string
            end_date: End date string
            data_cols: List of data columns to retrieve
            facets: Dictionary of facets to filter by
            offset: Pagination offset
            length: Number of results to return
            
        Returns:
            pd.DataFrame: Data returned from the API, or None if error/empty
        """
        if not self.api_key:
             raise ValueError("API Key is missing")

        url = f"{self.base_url}/{route}/data"
        
        params = {
            "api_key": self.api_key,
            "frequency": frequency,
            "start": start_date,
            "end": end_date,
            "offset": offset,
            "length": length,
        }
        
        for i, col in enumerate(data_cols):
            params[f"data[{i}]"] = col
            
        if facets:
            for key, value in facets.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        params[f"facets[{key}][{i}]"] = v
                else:
                    params[f"facets[{key}][]"] = value

        try:
            logger.info(f"Fetching data from {url}...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            payload = response.json()
            if "response" in payload and "data" in payload["response"]:
                data = payload["response"]["data"]
                if not data:
                    logger.warning("No data returned from API.")
                    return None
                return pd.DataFrame(data)
            else:
                logger.error(f"Unexpected API response format: {payload.keys()}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if response is not None:
                logger.error(f"Response: {response.text}")
            return None

if __name__ == "__main__":
    # Basic test execution
    logging.basicConfig(level=logging.INFO)
    eia = EIAData()
    print("EIAData initialized. Set EIA_API_KEY and call get_data() to test.")
