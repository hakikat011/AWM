"""
LLM inference engine for market intelligence analysis.
Handles model loading, prompt engineering, and inference operations.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

from .config import config

logger = logging.getLogger(__name__)


class LLMEngine:
    """LLM inference engine for market intelligence."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.inference_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the LLM model and tokenizer."""
        try:
            logger.info(f"Initializing LLM engine with model: {config.llm.model_name}")
            
            # Set device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU")
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.llm.model_name,
                cache_dir=config.llm.model_path,
                trust_remote_code=config.llm.trust_remote_code
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm.model_name,
                cache_dir=config.llm.model_path,
                torch_dtype=getattr(torch, config.llm.dtype),
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=config.llm.trust_remote_code,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available():
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(config.llm.gpu_memory_fraction)
                logger.info(f"GPU memory fraction set to: {config.llm.gpu_memory_fraction}")
            
            self.model_loaded = True
            logger.info("LLM engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response from LLM."""
        if not self.model_loaded:
            raise RuntimeError("LLM engine not initialized")
        
        max_tokens = max_tokens or config.llm.max_tokens
        temperature = temperature or config.llm.temperature
        
        async with self.inference_lock:
            try:
                start_time = time.time()
                
                # Format prompt with system message if provided
                if system_prompt:
                    formatted_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
                else:
                    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                
                # Tokenize input
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096 - max_tokens
                ).to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=config.llm.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                inference_time = time.time() - start_time
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    "response": response,
                    "inference_time": inference_time,
                    "tokens_generated": len(outputs[0]) - inputs.input_ids.shape[1],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error during LLM inference: {e}")
                raise
    
    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze sentiment of text with market context."""
        system_prompt = """You are an expert financial analyst specializing in Indian equity markets (NSE/BSE). 
Analyze the sentiment of the provided text in the context of Indian stock market trading.

Respond with a JSON object containing:
- sentiment: "positive", "negative", or "neutral"
- confidence: float between 0.0 and 1.0
- reasoning: brief explanation of the sentiment analysis
- market_impact: "bullish", "bearish", or "neutral"
- key_factors: list of key factors influencing the sentiment

Consider Indian market context, INR currency, SEBI regulations, and local economic factors."""
        
        prompt = f"Text to analyze: {text}"
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        result = await self.generate_response(prompt, system_prompt=system_prompt, max_tokens=512)
        
        try:
            # Try to parse JSON response
            response_text = result["response"]
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            parsed_response = json.loads(response_text)
            parsed_response["inference_time"] = result["inference_time"]
            return parsed_response
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "reasoning": "Failed to parse LLM response",
                "market_impact": "neutral",
                "key_factors": [],
                "raw_response": result["response"],
                "inference_time": result["inference_time"]
            }
    
    async def detect_market_regime(
        self,
        market_data: Dict[str, Any],
        lookback_period: int = 30
    ) -> Dict[str, Any]:
        """Detect current market regime."""
        system_prompt = """You are an expert quantitative analyst specializing in Indian equity markets.
Analyze the provided market data to determine the current market regime.

Respond with a JSON object containing:
- regime_type: "bull_market", "bear_market", "sideways", "high_volatility", "low_volatility"
- confidence: float between 0.0 and 1.0
- explanation: detailed explanation of the regime classification
- key_indicators: list of key indicators supporting the classification
- duration_estimate: estimated duration in days
- risk_level: "low", "medium", "high"

Consider Indian market characteristics, volatility patterns, and economic indicators."""
        
        prompt = f"Market data for analysis:\n{json.dumps(market_data, indent=2)}\nLookback period: {lookback_period} days"
        
        result = await self.generate_response(prompt, system_prompt=system_prompt, max_tokens=1024)
        
        try:
            response_text = result["response"]
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            parsed_response = json.loads(response_text)
            parsed_response["inference_time"] = result["inference_time"]
            return parsed_response
            
        except json.JSONDecodeError:
            return {
                "regime_type": "sideways",
                "confidence": 0.5,
                "explanation": "Failed to parse LLM response",
                "key_indicators": [],
                "duration_estimate": 30,
                "risk_level": "medium",
                "raw_response": result["response"],
                "inference_time": result["inference_time"]
            }
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.model_loaded = False
        logger.info("LLM engine cleaned up")


# Global LLM engine instance
llm_engine = LLMEngine()
