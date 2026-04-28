import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.features import encode_seasonality

class WeatherSimulator:
    """Generates realistic synthetic weather features for Gujarat."""
    
    @staticmethod
    def simulate_day(month, lat, lon):
        # 1. Base Seasonal Patterns (sine/cos)
        sin, cos = encode_seasonality(month)
        
        # 2. Temperature Logic
        # Hotter in summer (May/June), cooler in winter (Jan)
        # Lat/Lon impact: North (higher lat) is hotter/drier
        base_temp = 32 + (10 * sin) + (lat - 22) * 0.5
        temp = base_temp + np.random.normal(0, 2)
        
        # 3. Humidity Logic
        # High in monsoon (July/Aug), low in summer
        base_humid = 60 - (30 * sin) + (23 - lat) * 2
        humid = np.clip(base_humid + np.random.normal(0, 5), 20, 100)
        
        # 4. Pressure (hPa)
        # Low pressure in monsoon storms
        base_press = 1012 + (5 * cos) 
        press = base_press + np.random.normal(0, 1.5)
        
        # 5. Wind & Clouds (Correlated with Humidity)
        wind = max(5, 12 + (10 * sin) + np.random.normal(0, 3))
        clouds = np.clip(base_humid * 1.1 + np.random.normal(0, 10), 0, 100)
        
        return {
            "temperature": round(float(temp), 2),
            "humidity": round(float(humid), 2),
            "pressure": round(float(press), 2),
            "wind_speed": round(float(wind), 2),
            "cloud_cover": round(float(clouds), 2)
        }

class SimulationPipeline:
    """Orchestrates the 4-stage prediction flow."""
    
    def __init__(self, rainfall_model, flood_model):
        self.r_model = rainfall_model
        self.f_model = flood_model
        
    def run_simulation(self, city_meta, start_date, days=30, historical_seed=None):
        """
        city_meta: dict with lat, lon, elevation, river_distance
        historical_seed: list of rainfall [day-7, day-6, ..., day-1]
        """
        results = []
        
        # Seed memory (7 days)
        if historical_seed is None or len(historical_seed) < 7:
            rain_memory = [0.0] * 7
        else:
            rain_memory = historical_seed[-7:]
            
        current_date = start_date
        
        for _ in range(days):
            month = current_date.month
            
            # STAGE 1: Weather Simulation
            weather = WeatherSimulator.simulate_day(month, city_meta["lat"], city_meta["lon"])
            
            # STAGE 2: Rainfall Prediction
            # Model features: [month, temp, humid, pres, wind, cloud]
            atmos_features = np.array([[
                float(month),
                weather["temperature"], 
                weather["humidity"], 
                weather["pressure"], 
                weather["wind_speed"], 
                weather["cloud_cover"]
            ]])
            pred_rain = max(0.0, float(self.r_model.predict(atmos_features)[0]))
            
            # STAGE 3: Rolling Feature Engineering
            rain_memory.append(pred_rain)
            rain3 = sum(rain_memory[-3:])
            rain7 = sum(rain_memory[-7:])
            
            # STAGE 4: Advanced Flood Prediction
            # Model features: [rain, elev, dist, lat, lon, rain3, rain7, month]
            flood_features = np.array([[
                pred_rain, 
                city_meta["elevation"], 
                city_meta["river_distance"], 
                city_meta["lat"], 
                city_meta["lon"],
                rain3,
                rain7,
                month
            ]])
            
            # Get probability from XGBoost
            proba = float(self.f_model.predict_proba(flood_features)[0][1])
            
            results.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "rain_mm": round(pred_rain, 2),
                "flood_probability": round(proba, 3),
                "rain3": round(rain3, 2),
                "rain7": round(rain7, 2),
                **weather
            })
            
            # Increment day
            current_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
            
        return pd.DataFrame(results)

class DeepSimulationPipeline:
    """Orchestrates the DL-based 2-stage prediction flow (Phase 2)."""
    
    def __init__(self, rain_model, rain_scaler, flood_model, flood_scaler):
        self.rain_model = rain_model
        self.rain_scaler = rain_scaler
        self.flood_model = flood_model
        self.flood_scaler = flood_scaler
        
    def run_simulation(self, city_meta, start_date, days=30, historical_seed=None, population=1000000):
        import torch
        results = []
        
        # Seed memory (7 days)
        if historical_seed is None or len(historical_seed) < 7:
            rain_memory = [0.0] * 7
        else:
            rain_memory = historical_seed[-7:]
            
        current_date = start_date
        
        for _ in range(days):
            rain3 = sum(rain_memory[-3:])
            
            # STAGE 1: Rainfall Prediction (TabTransformer)
            # Features: [elev, dist, lat, lon, pop, rain3]
            r_feat = [city_meta["elevation"], city_meta["river_distance"], city_meta["lat"], city_meta["lon"], float(population), rain3]
            r_scaled = self.rain_scaler.transform([r_feat])
            
            with torch.no_grad():
                pred_rain = max(0.0, self.rain_model(torch.tensor(r_scaled, dtype=torch.float32)).item())
            
            # STAGE 2: Flood Prediction (FloodDNN)
            # Features: [rain, elev, dist, lat, lon]
            f_feat = [pred_rain, city_meta["elevation"], city_meta["river_distance"], city_meta["lat"], city_meta["lon"]]
            f_scaled = self.flood_scaler.transform([f_feat])
            
            with torch.no_grad():
                flood_proba = self.flood_model(torch.tensor(f_scaled, dtype=torch.float32)).item()
            
            rain_memory.append(pred_rain)
            
            results.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "rain_mm": round(pred_rain, 2),
                "flood_probability": round(flood_proba, 3),
                "rain3": round(rain3, 2),
                "type": "Deep Learning Simulation 🧠"
            })
            
            # Increment day
            current_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
            
        return pd.DataFrame(results)
