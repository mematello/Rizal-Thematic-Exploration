import re
from typing import Dict, List, Optional, Tuple

class TagalogRoleParser:
    def __init__(self):
        # Regex patterns for markers
        self.MARKER_AGENT = r'\b(ni|ng)\b' # Agent or Possessor
        self.MARKER_TOPIC = r'\b(si|ang)\b' # Subject/Focus (often Patient in passive)
        self.MARKER_OBLIQUE = r'\b(kay|sa)\b' # Indirect Object/Location
        
        # Simple verb detector (imperfect, identifying common affixes)
        self.VERB_AFFIXES = [
            r'^mag', r'^um', r'^in', r'^i-', r'^na', r'^ma', 
            r'um$', r'in$', r'hin$', r'an$'
        ]

    def _is_verb(self, word: str) -> bool:
        # Heuristic check for verbs
        if len(word) < 3:
            return False
        word = word.lower()
        for affix in self.VERB_AFFIXES:
            if re.search(affix, word):
                return True
        return False

    def parse_sentence(self, text: str) -> Dict[str, List[str]]:
        """
        Parses a Tagalog sentence into roles: Event, Agent, Patient, Oblique.
        Uses heuristics based on VSO/VOS structure and markers.
        """
        tokens = text.strip().split()
        if not tokens:
            return {}

        roles = {
            'event': [],
            'agent': [],
            'patient': [],
            'oblique': []
        }

        # Naive approach: Scan for markers and assign subsequent noun phrases
        # This is a rule-based approximation.
        
        current_role = None
        
        # Check if first word is likely a verb (Predicate-initial)
        first_word = tokens[0]
        # Clean punctuation
        first_word_clean = re.sub(r'[^\w\s]', '', first_word)
        
        if self._is_verb(first_word_clean):
            roles['event'].append(first_word_clean)
            start_idx = 1
        else:
            # Nominal predicate or topic-first? Assume predicate if not a marker.
            if not re.match(self.MARKER_TOPIC, first_word.lower()) and \
               not re.match(self.MARKER_AGENT, first_word.lower()) and \
               not re.match(self.MARKER_OBLIQUE, first_word.lower()):
                 roles['event'].append(first_word_clean)
                 start_idx = 1
            else:
                 start_idx = 0

        # Iterate through remaining tokens
        i = start_idx
        while i < len(tokens):
            token = tokens[i]
            word_clean = re.sub(r'[^\w\s]', '', token)
            word_lower = token.lower()
            
            # Check markers
            if re.match(self.MARKER_AGENT, word_lower):
                current_role = 'agent'
            elif re.match(self.MARKER_TOPIC, word_lower):
                current_role = 'patient' # Defaulting Topic to Patient for semantic search purposes (often true in Tagalog passive voice which is dominant)
            elif re.match(self.MARKER_OBLIQUE, word_lower):
                current_role = 'oblique'
            else:
                # Content word
                if current_role:
                    roles[current_role].append(word_clean)
                elif not roles['event']: # If we skipped first word as event but it wasn't
                     roles['event'].append(word_clean)
                else:
                     # If no role assigned yet, maybe part of the event phrase or default to patient
                     pass
            i += 1
            
        return roles

    def structured_string(self, text: str) -> str:
        """
        Returns a structured string for embedding:
        "EVENT: <val> AGENT: <val> PATIENT: <val> OBLIQUE: <val>"
        """
        parsed = self.parse_sentence(text)
        parts = []
        
        if parsed['event']:
            parts.append(f"EVENT: {' '.join(parsed['event'])}")
        if parsed['agent']:
            parts.append(f"AGENT: {' '.join(parsed['agent'])}")
        if parsed['patient']:
            parts.append(f"PATIENT: {' '.join(parsed['patient'])}")
        if parsed['oblique']:
            parts.append(f"OBLIQUE: {' '.join(parsed['oblique'])}")
            
        return " ".join(parts)

if __name__ == "__main__":
    # Simple test
    p = TagalogRoleParser()
    print(p.structured_string("Pinatay ni Basilio si Padre Damaso"))
    print(p.structured_string("Kamatayan ni Basilio"))
    print(p.structured_string("Ibinigay kay Maria ang sulat"))
