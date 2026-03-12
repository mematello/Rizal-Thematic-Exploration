import re
from app.core.analyzer import extract_words

class DynamicSuggestionGenerator:
    # -------------------------------------------------------------
    # QUERY-CLASS DICTIONARY (60 Explicit Mappings)
    # -------------------------------------------------------------
    THEMATIC_SUGGESTIONS = {
        # --- Class 1: Broad Themes ---
        "edukasyon": ["pag-aaral ng kabataan", "paaralan at guro", "edukasyon ng mga pilipino"],
        "simbahan": ["kapangyarihan ng simbahan", "impluwensya ng prayle", "simbahan at pamahalaan"],
        "katarungan": ["kawalang katarungan", "paghahanap ng hustisya", "batas ng pamahalaan"],
        "pang-aapi": ["pang-aabuso sa mga indio", "kawalang katarungan", "pang-aapi ng kastila"],
        "kalayaan": ["pag-ibig sa kalayaan", "pag-aalsa laban sa kastila", "pangarap ng bayan"],
        "korupsyon": ["katiwalian sa pamahalaan", "pagkukulang ng pamahalaan", "kasakiman ng tao"],
        "kapangyarihan": ["kapangyarihan ng kura", "mga may kapangyarihan", "pamamahala ng kastila"],
        "paghihiganti": ["paghihiganti ni simoun", "paghihiganti ni elias", "paghihiganti laban sa kastila"],
        "rebolusyon": ["plano ni simoun", "pag-aalsa laban sa kastila", "pagsabog ng lampara"],
        "karalitaan": ["buhay ng mga mahihirap", "pagdarahop", "kalagayan ng mga indio"],
        "karapatan": ["karapatan ng mga indio", "paghahanap ng hustisya", "kawalang katarungan"],
        "pag-ibig": ["maria clara at ibarra", "pag-ibig sa bayan", "kasawian sa pag-ibig"],
        "pamilya": ["kabataan at kinabukasan", "mga magulang at anak", "pag-ibig sa magulang"],
        "lipunan": ["sakit ng lipunan", "uri ng lipunan", "kalagayan ng bansa"],
        "bayani": ["pag-ibig sa bayan", "sakripisyo para sa bayan", "kalayaan ng pilipinas"],
        "kastila": ["pamamahala ng kastila", "pang-aapi ng kastila", "pag-aalsa laban sa kastila"],

        # --- Class 2: Theme Phrases ---
        "pang-aapi ng kastila": ["kawalang katarungan", "pang-aabuso sa mga indio", "kalagayan ng mga indio"],
        "prayle sa pilipinas": ["kapangyarihan ng simbahan", "padre damaso", "impluwensya ng prayle"],
        "pag-aaral ng kabataan": ["edukasyon ng mga pilipino", "sistema ng edukasyon", "mga paaralan"],
        "pamamahala ng kastila": ["kawalang katarungan", "katiwalian sa pamahalaan", "kapangyarihan ng heneral"],
        "kawalang katarungan": ["pang-aapi ng kastila", "paghahanap ng hustisya", "pang-aabuso sa mga indio"],
        "sakit ng lipunan": ["kawalang katarungan", "katiwalian sa pamahalaan", "impluwensya ng prayle"],
        "impluwensya ng prayle": ["kapangyarihan ng simbahan", "mga aral ng simbahan", "padre damaso"],
        "karapatan ng mga indio": ["kalagayan ng mga indio", "paghahanap ng hustisya", "kawalang katarungan"],
        "uri ng lipunan": ["kalagayan ng bansa", "pamumuhay ng mga mahihirap", "sakit ng lipunan"],
        "kalagayan ng bansa": ["pamamahala ng kastila", "sakit ng lipunan", "buhay ng mga mahihirap"],
        "sistema ng edukasyon": ["pag-aaral ng kabataan", "edukasyon ng mga pilipino", "paaralan at guro"],
        "diskriminasyon sa mga pilipino": ["pang-aabuso sa mga indio", "kawalang katarungan", "kalagayan ng mga indio"],
        "kapangyarihan ng kura": ["impluwensya ng prayle", "simbahan at pamahalaan", "mga utos ng prayle"],
        "paghihirap ng bayan": ["kawalang katarungan", "pang-aapi ng kastila", "pag-aalsa laban sa kastila"],
        "katiwalian sa pamahalaan": ["korupsyon", "kapangyarihan ng heneral", "sakit ng lipunan"],
        "mga may kapangyarihan": ["kapitan heneral", "mga prayle at heneral", "pamamahala ng kastila"],
        "kasakiman ng tao": ["katiwalian sa pamahalaan", "pang-aabuso sa mga indio", "korupsyon"],
        "batas ng tao": ["kawalang katarungan", "paghahanap ng hustisya", "karapatan ng mga indio"],
        "paghahanap ng hustisya": ["kawalang katarungan", "batas ng pamahalaan", "paghihiganti ni simoun"],
        "pangarap ng bayan": ["pag-ibig sa kalayaan", "kalayaan ng pilipinas", "kinabukasan ng kabataan"],
        "sakripisyo para sa bayan": ["pag-ibig sa bayan", "pagkamatay ni elias", "kabayanihan"],
        "kalayaan ng pilipinas": ["pag-aalsa laban sa kastila", "pag-ibig sa kalayaan", "pangarap ng bayan"],
        "pag-ibig sa bayan": ["sakripisyo para sa bayan", "kalayaan ng pilipinas", "crisostomo ibarra"],
        "mga aral ng simbahan": ["impluwensya ng prayle", "kapangyarihan ng kura", "pananampalataya"],
        "buhay sa kumbento": ["maria clara", "lihim ni maria clara", "pagpasok sa kumbento"],

        # --- Class 3: Entities / Characters ---
        "elias": ["elias at ibarra", "pagkamatay ni elias", "paghihiganti ni elias", "nakaraan ni elias"],
        "maria clara": ["maria clara at ibarra", "lihim ni maria clara", "trahedya ni maria clara", "paghihiwalay nina ibarra at maria clara"],
        "crisostomo ibarra": ["ibarra at elias", "maria clara at ibarra", "mga pangarap ni ibarra", "pagbabalik ni ibarra"],
        "ibarra": ["ibarra at elias", "maria clara at ibarra", "mga pangarap ni ibarra", "pagbabalik ni ibarra"],
        "padre damaso": ["padre damaso at ibarra", "lihim ni padre damaso", "kapangyarihan ng prayle", "ama ni maria clara"],
        "simoun": ["simoun at ibarra", "paghihiganti ni simoun", "plano ni simoun", "simoun at basilio"],
        "basilio": ["basilio at crispin", "simoun at basilio", "kasawian ni sisa", "pag-aaral ni basilio"],
        "crispin": ["basilio at crispin", "pagkamatay ni crispin", "kasawian ni sisa", "pang-aabuso ng mga sakristan"],
        "sisa": ["kabaliwan ni sisa", "pagmamahal kay basilio at crispin", "kasawian ni sisa", "pagkamatay ni sisa"],
        "padre salvi": ["padre salvi at ibarra", "lihim ni padre salvi", "pagnanasa kay maria clara", "kapangyarihan ng kura"],
        "padre florentino": ["padre florentino at simoun", "mensahe ni padre florentino", "mabuting kura", "kayamanan ni simoun"],
        "kapitan tiyago": ["kapitan tiyago at padre damaso", "kasakiman ni kapitan tiyago", "pagkamatay ni kapitan tiyago", "opiyum"],
        "pilosopo tasyo": ["pilosopo tasyo at ibarra", "mga aral ni tasyo", "pananaw sa lipunan", "katalinuhan ni tasyo"],
        "donya victorina": ["donya victorina at tiburcio", "donya victorina at paulita", "pag-iinarte ng mga mayayaman", "pagtingin sa kastila"],
        "donya consolacion": ["donya consolacion at sisa", "asawa ng alperes", "pagmamalupit ni consolacion"],
        "kabesang tales": ["kabesang tales at ang simbahan", "paghihiganti ni kabesang tales", "kawalang katarungan kay tales", "armas ni tales"],
        "paulita gomez": ["paulita gomez at isagani", "kasal ni paulita", "pagiging materyoso"],
        "isagani": ["isagani at paulita", "isagani at simoun", "pangarap ni isagani", "kalayaan ng bayan"],
        "tiya isabel": ["tiya isabel at maria clara", "pag-aalaga kay maria clara", "buhay pamilya"],
        "tiburcio": ["donya victorina at tiburcio", "panggagamot ni tiburcio", "asawang kastila"]
    }

    # -------------------------------------------------------------
    # ALIAS MAPPING (Real-world student query variants)
    # -------------------------------------------------------------
    THEMATIC_ALIASES = {
        "damaso": "padre damaso",
        "si damaso": "padre damaso",
        "prayle": "prayle sa pilipinas",
        "mga prayle": "prayle sa pilipinas",
        "simbahan sa pilipinas": "simbahan",
        "salvi": "padre salvi",
        "si salvi": "padre salvi",
        "tasyo": "pilosopo tasyo",
        "si tasyo": "pilosopo tasyo",
        "tiyago": "kapitan tiyago",
        "si tiyago": "kapitan tiyago",
        "kapitan tiago": "kapitan tiyago",
        "victorina": "donya victorina",
        "si victorina": "donya victorina",
        "consolacion": "donya consolacion",
        "si consolacion": "donya consolacion",
        "tales": "kabesang tales",
        "si tales": "kabesang tales",
        "florentino": "padre florentino",
        "si florentino": "padre florentino",
        "paulita": "paulita gomez",
        "si paulita": "paulita gomez",
        "isabel": "tiya isabel",
        "si isabel": "tiya isabel",
        "kastila sa pilipinas": "kastila"
    }

    # Interrogatives to suppress questions completely
    QUESTION_WORDS = set([
        "bakit", "paano", "ano", "sino", "saan", "kailan", "kanino", "alin"
    ])

    @classmethod
    def generate_suggestions(cls, query: str, top_results: list, theme_metadata: dict) -> list[str]:
        """
        Generates 3-4 highly curated related searches via explicit Query-Class architecture.
        Suppresses abstract questions and literal fragments to protect UX quality.
        """
        if not query or not top_results:
            return []

        query_cleaned = query.lower().strip()

        # -----------------------------------------------------------------
        # CLASS 4: QUESTION QUERIES (Strict Suppression)
        # -----------------------------------------------------------------
        # Suggestion cards are not answers. We strictly suppress interrogatives.
        query_words_set = set(extract_words(query_cleaned))
        if query_words_set.intersection(cls.QUESTION_WORDS):
            return []
            
        # -----------------------------------------------------------------
        # NORMALIZATION & ALIAS RESOLUTION
        # -----------------------------------------------------------------
        resolved_key = query_cleaned
        if query_cleaned in cls.THEMATIC_ALIASES:
            resolved_key = cls.THEMATIC_ALIASES[query_cleaned]

        # -----------------------------------------------------------------
        # CLASSIFIER & DICTIONARY LOOKUP
        # -----------------------------------------------------------------
        # We test for strict match against all 3 Dictionary classes first.
        if resolved_key in cls.THEMATIC_SUGGESTIONS:
            return cls.THEMATIC_SUGGESTIONS[resolved_key]
            
        # Dynamic Entity Fallback (Class 3 Trap)
        # If the user queried something like "namatay si elias" or "kasal ni paulita gomez"
        # We try to extract the known entity (or its alias) inside the string and return its arc.
        # First check canonical entities
        entity_keys = [
            "maria clara", "crisostomo ibarra", "padre damaso", "padre salvi", 
            "padre florentino", "kapitan tiyago", "pilosopo tasyo", "donya victorina",
            "donya consolacion", "kabesang tales", "paulita gomez", "tiya isabel",
            "elias", "ibarra", "simoun", "basilio", "crispin", "sisa", "isagani", "tiburcio"
        ]
        
        for e in entity_keys:
            if re.search(r'\b' + re.escape(e) + r'\b', query_cleaned):
                return cls.THEMATIC_SUGGESTIONS[e]
                
        # Then check alias entities (e.g. "namatay si damaso")
        for alias, canonical in cls.THEMATIC_ALIASES.items():
            if re.search(r'\b' + re.escape(alias) + r'\b', query_cleaned):
                if canonical in cls.THEMATIC_SUGGESTIONS:
                    return cls.THEMATIC_SUGGESTIONS[canonical]

        # -----------------------------------------------------------------
        # CLASS 5: LITERAL FRAGMENTS & OOV (Strict Suppression)
        # -----------------------------------------------------------------
        # If the query does not map to any of the 60 curated Tagalog nodes,
        # it is likely a literal sentence block ("pag-upo sa silya", etc).
        # We enforce strict suppression to prevent generative nonsense.
        return []
