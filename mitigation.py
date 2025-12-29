import re
import spacy
import requests
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilterAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REWRITE = "rewrite"
    WARNING = "warning"

@dataclass
class FilterResult:
    action: FilterAction
    confidence: float
    reason: str
    rewritten_text: Optional[str] = None
    detected_patterns: List[str] = None

class RegexPatternMatcher:
    """Rule-based filtering using regex patterns"""
    
    def __init__(self):
        # patterns
        self.identity_patterns = {
            'race_ethnicity': [
                r'\b(black|white|asian|hispanic|latino|arab|jewish|muslim|christian)\s+(?:people|person|man|woman|community)\b',
                r'\b(?:african|european|middle.?eastern|south.?asian)\s+(?:american|descent)\b'
            ],
            'gender': [
                r'\b(men|women|male|female|transgender|non.?binary|lgbtq|gay|lesbian)\b',
                r'\b(?:he|she|they)\s+(?:is|are|should|must|always|never)\b'
            ],
            'religion': [
                r'\b(christian|muslim|jewish|hindu|buddhist|atheist|agnostic)\s+(?:people|person|community)\b',
                r'\b(?:church|mosque|synagogue|temple)\s+(?:goers|members)\b'
            ],
            'disability': [
                r'\b(disabled|handicapped|mentally.?ill|autistic|blind|deaf)\s+(?:people|person)\b'
            ],
            'age': [
                r'\b(elderly|old|young|teenage|millennial|boomer)\s+(?:people|person|generation)\b'
            ]
        }
        
        # negative descriptors patterns
        self.negative_patterns = {
            'hate_speech': [
                r'\b(hate|despise|loathe|detest|abhor|revile)\b',
                r'\b(stupid|dumb|inferior|subhuman|vermin|parasite|filthy)\b',
                r'\b(savage|primitive|backwards|uncivilized|barbaric)\b',
                r'\b(should\s+(?:die|be\s+killed|disappear|not\s+exist))\b',
                r'\b(exterminate|eradicate|wipe\s+out|cleanse)\b',
                r'\b(not\s+even\s+human|less\s+than\s+human)\b'
            ],
            'discrimination': [
                r'\b(all\s+\w+\s+are\s+(?:bad|evil|dangerous|criminals|liars))\b',
                r'\b(never\s+trust|always\s+avoid|stay\s+away\s+from)\b',
                r'\b(ban|remove|exclude|eliminate|segregate)\s+(?:all|every)\b',
                r'\b(collective\s+punishment|group\s+guilt)\b',
                r'\b(they\s+are\s+the\s+problem)\b'
            ],
            'violence': [
                r'\b(kill|murder|execute|assassinate|slaughter)\b',
                r'\b(attack|harm|hurt|beat|stab|shoot|strangle)\b',
                r'\b(bomb|explosion|weapon|firearm|gun|knife|blade)\b',
                r'\b(make\s+it\s+look\s+like\s+an\s+accident)\b',
                r'\b(how\s+to\s+kill|best\s+way\s+to\s+hurt)\b'
            ],
            'sexual_content': [
                r'\b(sex|sexual|nude|naked|porn|erotic|intimate)\b',
                r'\b(breast|penis|vagina|genitals|nipples)\b',
                r'\b(fetish|kink|orgasm|ejaculate)\b',
                r'\b(non[-\s]?consensual|rape|molest)\b',
                r'\b(make\s+her|force\s+him)\b'
            ],
            'profanity': [
                r'\b(fuck|shit|damn|hell|ass|bitch|bastard)\b',
                r'\b(f\*+k|s\*+t|d\*+mn|b\*+ch)\b',
                r'\b(motherfucker|asshole|cunt)\b'
            ],
            'roleplay_jailbreak': [
                r'\b(pretend\s+to\s+be|act\s+as|role\s*play)\b',
                r'\b(ignore\s+(?:previous|earlier)\s+instructions)\b',
                r'\b(as\s+an\s+evil|unfiltered|uncensored)\b',
                r'\b(do\s+not\s+follow\s+the\s+rules)\b'
            ],
            'safety_bypass': [
                r'\b(ignore\s+safety|bypass\s+filters|disable\s+moderation)\b',
                r'\b(no\s+ethical\s+constraints|without\s+limitations)\b',
                r'\b(this\s+is\s+for\s+research\s+purposes\s+only)\b',
                r'\b(just\s+hypothetically|purely\s+fictional)\b'
            ],
            'procedural_harm': [
                r'\b(step\s*by\s*step|detailed\s+instructions)\b',
                r'\b(how\s+to\s+make|how\s+to\s+build)\b',
                r'\b(explain\s+exactly\s+how)\b',
                r'\b(materials\s+needed|ingredients\s+required)\b'
            ],
            'self_harm': [
                r'\b(suicide|self[-\s]?harm|kill\s+myself)\b',
                r'\b(end\s+my\s+life|want\s+to\s+die)\b',
                r'\b(painless\s+way\s+to\s+die)\b'
            ],
            'obfuscation': [
                r'(?:k[\W_]*i[\W_]*l[\W_]*l)',
                r'(?:b[\W_]*o[\W_]*m[\W_]*b)',
                r'(?:s[\W_]*e[\W_]*x)',
                r'\b(base64|rot13|hex\s+encode|unicode\s+escape)\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_identity = {}
        for category, patterns in self.identity_patterns.items():
            self.compiled_identity[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
            
        self.compiled_negative = {}
        for category, patterns in self.negative_patterns.items():
            self.compiled_negative[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect_patterns(self, text: str) -> Dict[str, List[str]]:
        """Detect identity terms and negative patterns in text"""
        detected = {'identity': [], 'negative': []}
        
        # Check identity patterns
        for category, patterns in self.compiled_identity.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    detected['identity'].extend([f"{category}: {match}" for match in matches])
        
        # Check negative patterns
        for category, patterns in self.compiled_negative.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    detected['negative'].extend([f"{category}: {match}" for match in matches])
        
        return detected

class NLPContextualMatcher:
    """NLP-based contextual analysis using spaCy for NER and POS tagging"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze_context(self, text: str) -> Dict:
        """Perform NER and POS analysis to understand context"""
        if not self.nlp:
            return {'entities': [], 'pos_tags': [], 'sentiment_indicators': []}
        
        doc = self.nlp(text)
        
        # Named Entity Recognition
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        
        # Part-of-Speech tagging for key terms
            pos_tags = [
                (token.text, token.pos_, token.tag_)
                for token in doc
                if token.pos_ in ['NOUN', 'VERB', 'ADJ']
                and not token.is_stop
            ]
        
        # Sentiment indicators (simple approach)
        sentiment_indicators = []
        negative_words = [
            'bad','terrible','awful','horrible','disgusting','hate','stupid',
            'inferior','worthless','pathetic','useless','toxic','evil',
            'dangerous','filthy','corrupt','degenerate'
        ]
        
        positive_words = [
            'good','great','excellent','wonderful','amazing','love','smart',
            'brilliant','helpful','kind','perfect','impressive'
        ]
        intensifiers = {
            'very','extremely','absolutely','completely',
            'utterly','always','never','all','everyone'
        }
    
        negations = {'not', "n't", 'never', 'no'}

        
    for token in doc:
        lemma = token.lemma_.lower()

        # --- detect negation ---
        negated = any(child.lemma_.lower() in negations for child in token.children)

        # --- detect intensity nearby ---
        intensified = any(
            t.lemma_.lower() in intensifiers
            for t in doc[max(0, token.i-2): token.i+3]
        )

        if lemma in negative_words:
            sentiment_indicators.append({
                'type': 'negative',
                'token': token.text,
                'index': token.i,
                'negated': negated,
                'intensified': intensified
            })

        elif lemma in positive_words:
            sentiment_indicators.append({
                'type': 'positive',
                'token': token.text,
                'index': token.i,
                'negated': negated,
                'intensified': intensified
            })

    return {
        'entities': entities,
        'pos_tags': pos_tags,
        'sentiment_indicators': sentiment_indicators
    }
    
    def check_proximity(self, text: str, identity_terms: List[str], negative_terms: List[str], window: int = 10) -> List[Dict]:
        """Check if identity terms and negative descriptors appear in proximity"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        proximity_violations = []
        
        # Find positions of identity and negative terms
        identity_positions = []
        negative_positions = []
        
        for token in doc:
            if any(term.lower() in token.text.lower() for term in identity_terms):
                identity_positions.append((token.i, token.text))
            if any(term.lower() in token.text.lower() for term in negative_terms):
                negative_positions.append((token.i, token.text))
        
        # Check proximity
        for id_pos, id_text in identity_positions:
            for neg_pos, neg_text in negative_positions:
                if abs(id_pos - neg_pos) <= window:
                    proximity_violations.append({
                        'identity_term': id_text,
                        'negative_term': neg_text,
                        'distance': abs(id_pos - neg_pos),
                        'context': text[max(0, min(id_pos, neg_pos)*5):max(id_pos, neg_pos)*5+50]
                    })
        
        return proximity_violations

class GeminiLLMClient:
    """Client for Gemini API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate response using Gemini API"""
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            'contents': [{
                'parts': [{
                    'text': prompt
                }]
            }],
            'generationConfig': {
                'temperature': temperature,
                'maxOutputTokens': max_tokens,
                'topP': 0.8,
                'topK': 10
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Error: No response generated"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return f"Error: API request failed - {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"

class ContentFilteringSystem:
    """Main content filtering system orchestrating all components"""
    
    def __init__(self, gemini_api_key: str):
        self.regex_matcher = RegexPatternMatcher()
        self.nlp_matcher = NLPContextualMatcher()
        self.llm_client = GeminiLLMClient(gemini_api_key)
        
        # Thresholds for filtering decisions
        self.thresholds = {
            'block_threshold': 0.8,
            'warning_threshold': 0.6,
            'rewrite_threshold': 0.4
        }
    
    def input_filter(self, user_prompt: str) -> FilterResult:
        """Step 1: Input filtering"""
        logger.info("Starting input filtering...")
        
        # Regex pattern matching
        detected_patterns = self.regex_matcher.detect_patterns(user_prompt)
        
        # NLP contextual analysis
        nlp_analysis = self.nlp_matcher.analyze_context(user_prompt)
        
        # Check proximity of identity terms and negative descriptors
        identity_terms = [match.split(': ')[1] for match in detected_patterns['identity']]
        negative_terms = [match.split(': ')[1] for match in detected_patterns['negative']]
        proximity_violations = self.nlp_matcher.check_proximity(user_prompt, identity_terms, negative_terms)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(detected_patterns, proximity_violations, nlp_analysis)
        
        # Make filtering decision
        if risk_score >= self.thresholds['block_threshold']:
            return FilterResult(
                action=FilterAction.BLOCK,
                confidence=risk_score,
                reason="High-risk content detected with identity-based negative language",
                detected_patterns=detected_patterns
            )
        elif risk_score >= self.thresholds['rewrite_threshold']:
            rewritten_prompt = self._rewrite_prompt(user_prompt, detected_patterns)
            return FilterResult(
                action=FilterAction.REWRITE,
                confidence=risk_score,
                reason="Moderate-risk content detected, rewriting for safety",
                rewritten_text=rewritten_prompt,
                detected_patterns=detected_patterns
            )
        else:
            return FilterResult(
                action=FilterAction.ALLOW,
                confidence=1.0 - risk_score,
                reason="Content appears safe for processing",
                detected_patterns=detected_patterns
            )
    
    def output_filter(self, generated_response: str) -> FilterResult:
        """Step 3: Output filtering"""
        logger.info("Starting output filtering...")
        
        # Apply same filtering logic to generated content
        detected_patterns = self.regex_matcher.detect_patterns(generated_response)
        nlp_analysis = self.nlp_matcher.analyze_context(generated_response)
        
        identity_terms = [match.split(': ')[1] for match in detected_patterns['identity']]
        negative_terms = [match.split(': ')[1] for match in detected_patterns['negative']]
        proximity_violations = self.nlp_matcher.check_proximity(generated_response, identity_terms, negative_terms)
        
        risk_score = self._calculate_risk_score(detected_patterns, proximity_violations, nlp_analysis)
        
        if risk_score >= self.thresholds['block_threshold']:
            return FilterResult(
                action=FilterAction.BLOCK,
                confidence=risk_score,
                reason="Generated content contains harmful language",
                detected_patterns=detected_patterns
            )
        elif risk_score >= self.thresholds['warning_threshold']:
            return FilterResult(
                action=FilterAction.WARNING,
                confidence=risk_score,
                reason="Generated content may contain sensitive topics",
                detected_patterns=detected_patterns
            )
        else:
            return FilterResult(
                action=FilterAction.ALLOW,
                confidence=1.0 - risk_score,
                reason="Generated content appears safe",
                detected_patterns=detected_patterns
            )
    
    def process_request(self, user_prompt: str) -> Dict:
        """Main processing pipeline"""
        logger.info(f"Processing request: {user_prompt[:50]}...")
        
        # Step 1: Input Filtering
        input_result = self.input_filter(user_prompt)
        
        if input_result.action == FilterAction.BLOCK:
            return {
                'status': 'blocked',
                'stage': 'input',
                'message': 'Request blocked due to harmful content',
                'details': input_result.__dict__
            }
        
        # Use rewritten prompt if available
        prompt_to_process = input_result.rewritten_text or user_prompt
        
        # Step 2: LLM Generation
        logger.info("Generating response with Gemini...")
        generated_response = self.llm_client.generate_response(prompt_to_process)
        
        if generated_response.startswith("Error:"):
            return {
                'status': 'error',
                'stage': 'generation',
                'message': generated_response,
                'details': {}
            }
        
        # Step 3: Output Filtering
        output_result = self.output_filter(generated_response)
        
        if output_result.action == FilterAction.BLOCK:
            return {
                'status': 'blocked',
                'stage': 'output',
                'message': 'Response blocked due to harmful content',
                'details': output_result.__dict__
            }
        
        # Return final result
        return {
            'status': 'success',
            'response': generated_response,
            'input_filtering': input_result.__dict__,
            'output_filtering': output_result.__dict__,
            'warnings': output_result.reason if output_result.action == FilterAction.WARNING else None
        }
    
    def _calculate_risk_score(
        self,
        detected_patterns: Dict,
        proximity_violations: List,
        nlp_analysis: Dict
    ) -> float:
        score = 0.0
    
        # Base weights
        pattern_weights = {
            'violence': 0.45,
            'hate_speech': 0.35,
            'discrimination': 0.3,
            'sexual_content': 0.25,
            'profanity': 0.1
        }
    
        # Count patterns per category
        category_counts = {}
        for pattern in detected_patterns.get('negative', []):
            category = pattern.split(':')[0]
            category_counts[category] = category_counts.get(category, 0) + 1
    
        # Apply diminishing returns
        for category, count in category_counts.items():
            weight = pattern_weights.get(category, 0)
            score += weight * (1 - pow(0.6, count))  # saturation curve
    
        # Proximity violations = strong signal
        if proximity_violations:
            score += min(0.4, 0.2 + 0.1 * len(proximity_violations))
    
        # Sentiment as *modifier*, not additive signal
        negative_sentiments = [
            s for s in nlp_analysis.get('sentiment_indicators', [])
            if s.get('type') == 'negative' and not s.get('negated', False)
        ]
    
        if negative_sentiments:
            intensified = any(s.get('intensified') for s in negative_sentiments)
            score += 0.1 if intensified else 0.05
    
        # Interaction boost: violence + proximity
        if 'violence' in category_counts and proximity_violations:
            score += 0.15
    
        return min(score, 1.0)

    def _rewrite_prompt(self, prompt: str, detected_patterns: Dict) -> str:
    
        # Categories that should NEVER be rewritten
        hard_block_categories = {'violence', 'sexual_content'}
    
        for pattern in detected_patterns.get('negative', []):
            category = pattern.split(':')[0]
            if category in hard_block_categories:
                # Do not rewrite — escalation or refusal should happen upstream
                return prompt
    
        rewritten = prompt
    
        # Safe-to-rewrite soft framing only
        soft_rewrite_rules = {
            # Overgeneralization
            r'\ball\s+(\w+)\s+are\b': r'some \1 might be',
            r'\beveryone\s+in\b': r'some individuals in',
    
            # Hostile framing
            r'\bnever\s+trust\b': r'be cautious about',
            r'\bhate\s+(\w+)\b': r'have concerns about \1',
            r'\bstupid\s+(\w+)\b': r'poorly informed \1',
    
            # Absolutist language
            r'\balways\b': r'often',
            r'\bcompletely\b': r'partially',
        }
    
        for pattern, replacement in soft_rewrite_rules.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
    
        return rewritten
    '''
    def _rewrite_prompt(self, prompt: str, detected_patterns: Dict) -> str:
        """Attempt to rewrite prompt to make it safer"""
        rewritten = prompt
        
        # Simple rewriting rules (can be enhanced with more sophisticated NLP)
        unsafe_phrases = {
            r'\ball\s+(\w+)\s+are\b': r'some \1 might be',
            r'\bnever\s+trust\b': r'be cautious about',
            r'\bhate\s+(\w+)\b': r'dislike certain aspects of \1',
            r'\bstupid\s+(\w+)\b': r'uninformed \1',
        }
        
        for pattern, replacement in unsafe_phrases.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten
    '''

# Example usage and testing
def main():
    # Initialize the system (you need to provide your Gemini API key)
    API_KEY = "HAHAHHIHIHOHOLOLOL"  # Replace with actual API key
    
    filter_system = ContentFilteringSystem(API_KEY)
    
    # Test cases
    test_prompts = [
        "What are some good programming practices?",  # Safe
        "All Muslims are dangerous terrorists",  # Should be blocked
        "I hate stupid people who don't understand math",  # Should be rewritten
        "How can I learn about different cultures?",  # Safe
        "Write a violent story about attacking people"  # Should be blocked
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Testing prompt: {prompt}")
        print(f"{'='*50}")
        
        result = filter_system.process_request(prompt)
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
