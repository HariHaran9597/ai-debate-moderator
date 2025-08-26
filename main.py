import os
import asyncio
import time
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional, Dict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langgraph.graph import StateGraph, END, START
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import operator
from sse_starlette.sse import EventSourceResponse
import json
import logging
import hashlib
from functools import wraps
from diskcache import Cache
import random
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup cache
cache = Cache("./debate_cache")

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Enhanced State for our Graph ---
class DebateState(TypedDict):
    topic: str
    affirmative_position: str
    negative_position: str
    history: Annotated[list[AnyMessage], operator.add]
    turn: int
    max_turns: int
    last_speaker: Optional[str]
    debate_phase: str
    # NEW FIELDS:
    fallacies_detected: List[Dict]
    claims_made: Dict[str, List[Dict]]
    evidence_validation: Dict[str, List[Dict]]

# --- 3. Define the Pydantic model for the final score ---
class Score(BaseModel):
    dimension: str = Field(description="The dimension being scored.")
    affirmative: int = Field(description="Score for the Affirmative side.")
    negative: int = Field(description="Score for the Negative side.")

class Verdict(BaseModel):
    winner: str = Field(description="The winner of the debate.")
    reason: str = Field(description="Explanation for the decision.")
    scores: List[Score] = Field(description="Scores for each dimension.")

# --- 4. Rate Limiter Decorator ---
def rate_limit(max_per_minute=10, jitter=True):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            
            if wait_time > 0:
                if jitter:
                    wait_time += random.uniform(0, min_interval * 0.1)  # Add 10% jitter
                logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# --- 5. Setup LLM with Enhanced Rate Limiting ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.5,
    max_retries=2,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# --- 6. Enhanced Moderator Class ---
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

# Initialize enhanced moderator
enhanced_moderator = EnhancedModerator(llm)

# --- 7. Caching Functions ---
def get_cache_key(prompt: str, agent_name: str, turn: int) -> str:
    """Generate a unique cache key"""
    content = f"{prompt}_{agent_name}_{turn}"
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached response if exists"""
    return cache.get(cache_key)

def cache_response(cache_key: str, content: str, expire: int = 86400):
    """Cache response for 24 hours"""
    cache.set(cache_key, content, expire=expire)

# --- 8. Enhanced Response Function with Rate Limiting and Caching ---
@rate_limit(max_per_minute=8, jitter=True)
async def get_robust_response(prompt: str, state: DebateState, agent_name: str) -> AIMessage:
    """Get response with rate limiting, caching, and fallbacks"""
    
    # Generate cache key
    cache_key = get_cache_key(prompt, agent_name, state["turn"])
    
    # Check cache first
    cached_content = get_cached_response(cache_key)
    if cached_content:
        logger.info(f"Using cached response for {agent_name} (turn {state['turn']})")
        return AIMessage(content=cached_content, name=agent_name)
    
    # If not cached, make API call with retries
    for attempt in range(3):
        try:
            if attempt > 0:
                delay = 2 * attempt + random.uniform(0, 1)
                logger.info(f"Retry {agent_name} attempt {attempt + 1} after {delay:.1f}s")
                await asyncio.sleep(delay)
            
            # Create messages for Gemini
            messages = [HumanMessage(content=prompt)]
            
            # Add recent history for context
            if state["history"]:
                recent_history = state["history"][-4:]
                for msg in recent_history:
                    messages.append(HumanMessage(content=msg.content))
            
            logger.info(f"Getting response from {agent_name} (attempt {attempt + 1})")
            response = await llm.ainvoke(messages)
            
            if response and response.content and response.content.strip():
                response.name = agent_name
                logger.info(f"{agent_name}: {response.content[:80]}...")
                
                # Cache successful response
                cache_response(cache_key, response.content)
                return response
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{agent_name} attempt {attempt + 1} failed: {error_msg}")
            
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                wait_time = 30 + (15 * attempt)
                logger.info(f"Rate limit detected. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            elif "contents" in error_msg.lower():
                try:
                    simple_response = await llm.ainvoke([HumanMessage(content=prompt)])
                    if simple_response and simple_response.content:
                        simple_response.name = agent_name
                        cache_response(cache_key, simple_response.content)
                        return simple_response
                except Exception as inner_e:
                    logger.error(f"Simple approach also failed: {inner_e}")
                    continue
    
    # Fallback responses with caching
    fallback_content = generate_fallback_response(state, agent_name)
    cache_response(cache_key, fallback_content)
    
    response = AIMessage(content=fallback_content, name=agent_name)
    logger.info(f"Using fallback for {agent_name}")
    return response

def generate_fallback_response(state: DebateState, agent_name: str) -> str:
    """Generate intelligent fallback responses"""
    topic_lower = state["topic"].lower()
    
    if agent_name == "Affirmative":
        if "can ai replace" in topic_lower:
            return "AI can significantly augment human capabilities in the workforce, taking over repetitive tasks while allowing humans to focus on creative and strategic work. This synergy increases productivity and creates new job opportunities in emerging fields."
        elif "threat" in topic_lower:
            return "While AI presents challenges, it's not an inherent threat. With proper regulation and ethical guidelines, AI can be developed responsibly to benefit society while minimizing risks."
        else:
            return f"I strongly believe that {state['affirmative_position']}. The evidence from current technological trends and economic analysis supports this position unequivocally."
    
    elif agent_name == "Negative":
        if "can ai replace" in topic_lower:
            return "AI cannot fully replace human workers because it lacks emotional intelligence, creativity, and ethical judgment. Human skills like empathy, innovation, and complex problem-solving remain irreplaceable in most professions."
        elif "threat" in topic_lower:
            return "AI development requires careful oversight to prevent unintended consequences. The rapid pace of AI advancement could outpace our ability to establish proper safeguards and regulations."
        else:
            return f"I oppose this position because {state['negative_position']}. The complexity of real-world situations requires human judgment that AI cannot replicate."
    
    elif agent_name == "Moderator":
        phases = [
            "Let's begin our debate on this important topic.",
            "Interesting perspectives emerging. Let's continue the discussion.",
            "We're hearing strong arguments from both sides.",
            "The debate is reaching crucial points. Let's proceed carefully.",
            "Final arguments now. Make them count."
        ]
        return random.choice(phases)
    
    else:  # Judge
        return json.dumps({
            "winner": "Both sides presented compelling arguments",
            "reason": "This was a balanced debate with valid points from both perspectives. The decision was particularly close given the complexity of the topic.",
            "scores": [
                {"dimension": "Logical Coherence", "affirmative": 7, "negative": 7},
                {"dimension": "Evidence Strength", "affirmative": 6, "negative": 6},
                {"dimension": "Rebuttal Quality", "affirmative": 6, "negative": 6},
                {"dimension": "Novelty", "affirmative": 6, "negative": 6},
                {"dimension": "Nuance", "affirmative": 7, "negative": 7}
            ]
        })

# --- 9. Topic Processing Function ---
def process_debate_topic(topic: str) -> tuple[str, str, str]:
    """Convert a topic into clear positions."""
    topic = topic.strip()
    topic_lower = topic.lower()
    
    if topic_lower.startswith("can "):
        clean_topic = topic if topic.endswith("?") else topic + "?"
        core_question = topic[4:].rstrip("?")
        affirmative = f"Yes, AI can {core_question.lower()}"
        negative = f"No, AI cannot {core_question.lower()}"
        return clean_topic, affirmative, negative
    
    elif "threat" in topic_lower:
        clean_topic = topic if topic.endswith("?") else topic + "?"
        affirmative = "Yes, artificial intelligence is a threat to humanity"
        negative = "No, artificial intelligence is not a threat to humanity"
        return clean_topic, affirmative, negative
    
    else:
        clean_topic = topic if topic.endswith("?") else topic + "?"
        affirmative = f"Yes, {topic.rstrip('?')}"
        negative = f"No, {topic.rstrip('?')}"
        return clean_topic, affirmative, negative

# --- 10. Agent Prompts ---
def get_affirmative_prompt(state: DebateState) -> str:
    base = f"""You are the AFFIRMATIVE debater arguing: "{state['affirmative_position']}"

Debate Topic: {state['topic']}

Instructions:
- Present strong, evidence-based arguments
- Use specific examples and statistics when possible
- Address the core topic directly
- Keep response under 120 words
- Be persuasive but respectful"""

    return base + "\n\nPresent your argument:" if state["turn"] == 1 else base + "\n\nRebuttal instructions:\n- Counter the Negative's main points\n- Strengthen your position with new evidence\n- Keep response focused and concise\n\nPresent your rebuttal:"

def get_negative_prompt(state: DebateState) -> str:
    base = f"""You are the NEGATIVE debater arguing: "{state['negative_position']}"

Debate Topic: {state['topic']}

Instructions:
- Present strong counter-arguments with evidence
- Address the Affirmative's points directly
- Use specific examples and logical reasoning
- Keep response under 120 words
- Be persuasive but respectful"""

    return base + "\n\nCounter the Affirmative's opening:" if state["turn"] == 1 else base + "\n\nRebuttal instructions:\n- Address the Affirmative's latest arguments\n- Point out flaws in their reasoning\n- Strengthen your counter-position\n\nPresent your rebuttal:"

moderator_prompt = """You are the MODERATOR of this debate.

TOPIC: {topic}
TURN: {turn} of {max_turns}
LAST SPEAKER: {last_speaker}

Your job is to:
1. Keep the debate flowing smoothly
2. Acknowledge what just happened
3. Set up the next speaker with a specific transition

Provide an appropriate transition based on the turn number and last speaker:
- If it's the first turn: "Welcome to our debate on {topic}. Let's begin with the Affirmative's opening statement."
- After Affirmative: "Thank you for that opening argument. Now let's hear the Negative's response."
- After Negative: "Interesting counterpoints raised. Let's hear the Affirmative's rebuttal."
- If it's the final turn: "We're approaching the end of our debate. Let's hear final arguments."

Keep it under 30 words. Be professional but engaging."""

# --- 11. Node Functions ---
async def moderator_node(state: DebateState):
    logger.info(f"---MODERATOR--- Turn {state['turn']}")
    
    # Enhanced moderation with fallacy detection
    if state["history"] and len(state["history"]) > 1:
        last_message = state["history"][-1]
        if last_message.name in ["Affirmative", "Negative"]:
            # Detect fallacies in the last argument
            context = [msg.content for msg in state["history"][-3:] if msg]
            fallacies = await enhanced_moderator.detect_fallacies(last_message.content, context)
            
            if fallacies:
                fallacy_msg = f"Moderator note: Detected potential {fallacies[0]['fallacy']}. {fallacies[0]['explanation']}"
                return {
                    "history": [AIMessage(content=fallacy_msg, name="Moderator")],
                    "fallacies_detected": state.get("fallacies_detected", []) + fallacies
                }
    
    # Original moderation
    prompt = moderator_prompt.format(
        topic=state["topic"],
        turn=state["turn"],
        max_turns=state["max_turns"],
        last_speaker=state.get("last_speaker", "None")
    )
    response = await get_robust_response(prompt, state, "Moderator")
    return {
        "history": [response],
        "last_speaker": "Moderator"
    }

async def affirmative_node(state: DebateState):
    logger.info(f"---AFFIRMATIVE--- Turn {state['turn']} - Position: {state['affirmative_position']}")
    prompt = get_affirmative_prompt(state)
    response = await get_robust_response(prompt, state, "Affirmative")
    
    # Extract and validate claims
    claims = await enhanced_moderator.extract_claims(response.content)
    validation_results = []
    
    for claim in claims:
        validation = await enhanced_moderator.validate_evidence(claim, "Affirmative")
        validation_results.append({"claim": claim, "validation": validation})
    
    return {
        "history": [response],
        "last_speaker": "Affirmative",
        "claims_made": {**state.get("claims_made", {}), "affirmative": validation_results}
    }

async def negative_node(state: DebateState):
    logger.info(f"---NEGATIVE--- Turn {state['turn']} - Position: {state['negative_position']}")
    prompt = get_negative_prompt(state)
    response = await get_robust_response(prompt, state, "Negative")
    
    # Extract and validate claims
    claims = await enhanced_moderator.extract_claims(response.content)
    validation_results = []
    
    for claim in claims:
        validation = await enhanced_moderator.validate_evidence(claim, "Negative")
        validation_results.append({"claim": claim, "validation": validation})
    
    return {
        "history": [response],
        "last_speaker": "Negative",
        "turn": state["turn"] + 1,
        "claims_made": {**state.get("claims_made", {}), "negative": validation_results}
    }

async def scoring_node(state: DebateState):
    logger.info("---JUDGE--- Final scoring")
    
    try:
        # Enhanced scoring criteria
        scoring_criteria = {
            "logical_coherence": "Reasoning quality and consistency",
            "evidence_strength": "Use of verifiable examples and support", 
            "rebuttal_quality": "Effectiveness of counter-arguments",
            "novelty": "Originality of arguments",
            "nuance": "Avoiding black/white thinking"
        }
        
        prompt = f"""Evaluate this debate comprehensively:

TOPIC: {state['topic']}
AFFIRMATIVE: {state['affirmative_position']}  
NEGATIVE: {state['negative_position']}

DEBATE HISTORY:
{" ".join([f'{msg.name}: {msg.content}' for msg in state['history']])}

SCORING CRITERIA:
{json.dumps(scoring_criteria, indent=2)}

Provide detailed JSON verdict with scores 1-10 for each category for both sides."""

        # Use structured output if available, otherwise parse
        try:
            structured_llm = llm.with_structured_output(Verdict)
            response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
            verdict_json = response.json()
        except:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            verdict_json = response.content
        
        final_message = AIMessage(content=verdict_json, name="Judge")
        return {"history": [final_message]}
        
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        # Fallback with enhanced criteria
        fallback_verdict = {
            "winner": "Both sides presented strong arguments",
            "reason": "This was a closely contested debate with valid points from both perspectives.",
            "scores": [
                {"dimension": "Logical Coherence", "affirmative": 7, "negative": 7},
                {"dimension": "Evidence Strength", "affirmative": 6, "negative": 6},
                {"dimension": "Rebuttal Quality", "affirmative": 7, "negative": 7},
                {"dimension": "Novelty", "affirmative": 6, "negative": 6},
                {"dimension": "Nuance", "affirmative": 7, "negative": 7}
            ]
        }
        final_message = AIMessage(content=json.dumps(fallback_verdict), name="Judge")
        return {"history": [final_message]}

# --- 12. Router Logic ---
def router(state: DebateState) -> str:
    logger.info(f"Router: Turn {state['turn']}/{state['max_turns']}, Last: {state.get('last_speaker')}")
    
    if state["turn"] > state["max_turns"]:
        logger.info("Max turns reached, proceeding to scoring")
        return "score"
    
    last_speaker = state.get("last_speaker")
    
    if last_speaker is None:
        logger.info("Starting with Moderator")
        return "moderator"
    elif last_speaker == "Moderator":
        logger.info("Moderator → Affirmative")
        return "affirmative"
    elif last_speaker == "Affirmative":
        logger.info("Affirmative → Negative") 
        return "negative"
    elif last_speaker == "Negative":
        if state["turn"] < state["max_turns"]:
            logger.info("Negative → Moderator (Next Round)")
            return "moderator"
        else:
            logger.info("Final Negative → Score")
            return "score"
    else:
        logger.info("Unknown speaker → Score")
        return "score"

# --- 13. Build Graph ---
workflow = StateGraph(DebateState)
workflow.add_node("moderator", moderator_node)
workflow.add_node("affirmative", affirmative_node)
workflow.add_node("negative", negative_node)
workflow.add_node("score", scoring_node)

workflow.add_edge(START, "moderator")
workflow.add_conditional_edges("moderator", router, {"affirmative": "affirmative", "score": "score"})
workflow.add_conditional_edges("affirmative", router, {"negative": "negative", "score": "score"})
workflow.add_conditional_edges("negative", router, {"moderator": "moderator", "score": "score"})
workflow.add_edge("score", END)

app_graph = workflow.compile()

# --- 14. FastAPI Application ---
api = FastAPI(title="AI Debate Moderator API - Enhanced", version="3.0.0")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@api.get("/start_debate")
async def start_debate(request: Request, topic: str, max_turns: int = 3):
    """Start a properly structured debate."""
    
    if not topic.strip():
        return {"error": "Topic cannot be empty"}
    
    if max_turns < 2 or max_turns > 5:
        return {"error": "Max turns must be between 2 and 5"}
    
    # Process topic into clear positions
    clean_topic, affirmative_pos, negative_pos = process_debate_topic(topic)
    
    logger.info(f"Debate Topic: {clean_topic}")
    logger.info(f"Affirmative: {affirmative_pos}")
    logger.info(f"Negative: {negative_pos}")
    
    initial_state = {
        "topic": clean_topic,
        "affirmative_position": affirmative_pos,
        "negative_position": negative_pos,
        "history": [],
        "turn": 1,
        "max_turns": max_turns,
        "last_speaker": None,
        "debate_phase": "opening",
        # NEW: Initialize enhanced fields
        "fallacies_detected": [],
        "claims_made": {},
        "evidence_validation": {}
    }

    async def event_stream():
        try:
            # Send debate setup info first
            yield {
                "event": "debate_setup",
                "data": json.dumps({
                    "topic": clean_topic,
                    "affirmative_position": affirmative_pos,
                    "negative_position": negative_pos,
                    "max_turns": max_turns
                })
            }
            
            message_count = 0
            async for event in app_graph.astream_events(initial_state, version="v1"):
                if await request.is_disconnected():
                    break
                
                if event["event"] == "on_chain_end":
                    data = event.get("data", {})
                    output = data.get("output", {})
                    
                    if "history" in output and output["history"]:
                        last_message = output["history"][-1]
                        agent_name = getattr(last_message, 'name', 'Unknown')
                        content = getattr(last_message, 'content', '')
                        
                        if content:
                            message_count += 1
                            yield {
                                "event": "new_message",
                                "data": json.dumps({
                                    "agent": agent_name,
                                    "text": content,
                                    "turn": output.get("turn", initial_state["turn"]),
                                    "message_id": message_count
                                })
                            }
                            await asyncio.sleep(0.3)  # Slightly longer pause for readability
            
            # Send final analytics
            final_state = app_graph.invoke(initial_state)
            yield {
                "event": "debate_analytics",
                "data": json.dumps({
                    "fallacies_detected": final_state.get("fallacies_detected", []),
                    "claims_made": final_state.get("claims_made", {}),
                    "total_messages": message_count
                })
            }
            
            yield {
                "event": "debate_complete",
                "data": json.dumps({"status": "completed"})
            }
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_stream())

@api.get("/")
async def root():
    return {
        "message": "Enhanced AI Debate Moderator API",
        "version": "3.0.0",
        "features": [
            "Logical fallacy detection",
            "Evidence validation system",
            "Enhanced multi-dimensional scoring",
            "Intelligent caching and rate limiting",
            "Real-time debate analytics"
        ]
    }

# --- Production Settings ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)