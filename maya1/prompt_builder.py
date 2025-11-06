"""
Maya1 Prompt Builder
Builds formatted prompts for description-conditioned TTS.
Format: <SOH><BOS><description="..."> text<EOT><EOH><SOA><SOS>
"""

from .constants import ALL_EMOTION_TAGS


class Maya1PromptBuilder:
    """Builds prompts in the format expected by Maya1 model."""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def build_prefix(self, description: str, text: str) -> str:
        # Format as: <description="..."> text
        formatted_text = f'<description="{description}"> {text}'
        # Build full prefix with special tokens
        prompt = (
            self.model.soh_token +
            self.model.bos_token +
            formatted_text +
            self.model.eot_token +
            self.model.eoh_token +
            self.model.soa_token +
            self.model.sos_token
        )
        
        return prompt
