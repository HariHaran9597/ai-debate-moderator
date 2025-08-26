import json
import aiohttp
import asyncio
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage

class EnhancedModerator:
    def __init__(self, llm):
        self.llm = llm
        self.fallacy_types = [
            "strawman", "ad hominem", "false dilemma", 
            "slippery slope", "circular reasoning", "hasty generalization"
        ]
    
    async def detect_fallacies(self, text: str, context: List[str]) -> List[Dict]:
        """Detect logical fallacies in arguments"""
        try:
            prompt = f"""Analyze this argument for logical fallacies:

ARGUMENT: {text}
CONTEXT: {context[-2:] if context else "No context"}

Identify any logical fallacies from: {", ".join(self.fallacy_types)}

Return JSON: [{{"fallacy": "type", "explanation": "brief reason", "confidence": "high/medium/low"}}]"""

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return json.loads(response.content)
        except:
            return []
    
    async def validate_evidence(self, claim: str, agent: str) -> Dict:
        """Evaluate evidence quality of claims"""
        try:
            prompt = f"""Evaluate this claim for evidence quality: "{claim}"

Consider:
1. Verifiability (high/medium/low)
2. Specificity (specific/vague)
3. Evidence type needed (statistics/studies/examples)

Return JSON: {{"verifiability": "rating", "specificity": "rating", "evidence_needed": "type"}}"""

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return json.loads(response.content)
        except:
            return {"verifiability": "unknown", "specificity": "unknown", "evidence_needed": "unknown"}
    
    async def extract_claims(self, text: str) -> List[str]:
        """Extract specific claims from text"""
        try:
            prompt = f"""Extract specific factual claims from this text:

TEXT: {text}

Return as JSON list of claims: ["claim1", "claim2", ...]"""

            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return json.loads(response.content)
        except:
            return []