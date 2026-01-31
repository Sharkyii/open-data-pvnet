import pytest
import pandas as pd
from unittest.mock import Mock, patch
from open_data_pvnet.scripts.fetch_eia_data import EIAData

@pytest.fixture
def mock_response():
    """Fixture to mock a successful API response."""
    mock = Mock()
    mock.json.return_value = {
        "response": {
            "data": [
                {"period": "2023-01-01T00", "value": 100, "fueltype": "SUN"},
                {"period": "2023-01-01T01", "value": 150, "fueltype": "SUN"},
            ]
        }
    }
    mock.raise_for_status.return_value = None
    return mock

def test_init_with_key():
    eia = EIAData(api_key="test_key")
    assert eia.api_key == "test_key"

def test_init_without_key(mocker):
    mocker.patch.dict("os.environ", {}, clear=True)
    eia = EIAData()
    assert eia.api_key is None

def test_get_data_success(mock_response):
    with patch("requests.get", return_value=mock_response) as mock_get:
        eia = EIAData(api_key="test_key")
        
        df = eia.get_data(
            route="test/route",
            frequency="hourly",
            start_date="2023-01-01",
            end_date="2023-01-02",
            data_cols=["value"],
            facets={"fueltype": "SUN"}
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "value" in df.columns
        
        # Verify API call parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["api_key"] == "test_key"
        assert kwargs["params"]["facets[fueltype][]"] == "SUN"
        assert kwargs["params"]["data[0]"] == "value"

def test_get_data_missing_key():
    eia = EIAData(api_key=None)
    with pytest.raises(ValueError, match="API Key is missing"):
        eia.get_data("route", "hourly", "start", "end")

def test_get_data_api_error():
    mock_resp = Mock()
    import requests
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error")
    
    with patch("requests.get", return_value=mock_resp):
        eia = EIAData(api_key="test_key")
        df = eia.get_data("route", "hourly", "start", "end")
        assert df is None

def test_get_data_empty_response():
    mock_resp = Mock()
    mock_resp.json.return_value = {"response": {"data": []}}
    mock_resp.raise_for_status.return_value = None
    
    with patch("requests.get", return_value=mock_resp):
        eia = EIAData(api_key="test_key")
        df = eia.get_data("route", "hourly", "start", "end")
        assert df is None
