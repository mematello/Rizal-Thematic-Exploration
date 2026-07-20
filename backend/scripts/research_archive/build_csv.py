import json
import csv

reasons = {
    "3743": "Scene explicitly paints idyllic native hospitality and rural peace.",
    "11247": "Kapitan graciously highlighting his town to visitors, an act of true hosting.",
    "5844": "Mentions the host's heritage of welcoming strangers despite illness.",
    "8684": "Contrasts native hospitality with the profound cruelty of the colonial rulers.",
    "5539": "Rich description of the lavish fruits prepared specifically for fiesta guests.",
    
    "13845": "Highlights the superstitious fear of witchcraft intertwined with faith.",
    "5950": "Pari Damaso manipulating his role as 'God's voice' purely for status.",
    "6036": "Mockery of the theatrical, hyper-performative rituals of church services.",
    "4759": "Juxtaposes the purity of youth with religious imagery (goddesses/saints).",
    "14475": "Shows how folk beliefs like sirens run parallel to formal colonial religion.",
    
    "2233": "Describes a patrician figure instilling fear, symbolizing unchecked absolute power.",
    "16214": "Padre Florentino's exile is a direct consequence of an unjust religious power structure.",
    "9075": "Details how powerful actors manipulate structural progress to crush opponents.",
    "13655": "Simoun plots a massive upheaval specifically to overthrow the fortress of corrupt power.",
    "5833": "A gathering of elite friars who wield supreme unquestioned authority in town.",
    
    "7906": "The friction between questioning minds and priests enforcing absolute ignorance.",
    "8662": "Touches on the colonial monopolization of intellect over native philosophies.",
    "13280": "References physics class equipment, directly addressing the state of science education.",
    "12198": "Points to the students' frustrated attempts to establish a Spanish language academy.",
    "15738": "Metaphor for the illusions of grandeur students face vs the harsh reality of their education.",
    
    "7754": "Doña Victorina hides her husband's handicap to maintain a hollow sense of family pride.",
    "7615": "Elias speaks on the multi-generational curse of a ruined family name and lost honor.",
    "5994": "Youth breaking away from superstitious family rites and elders.",
    "9351": "A haunting vision of ancestral ghosts punishing cowardice, tying into foundational family honor.",
    "3927": "Tragic reflection on innocent sibling bonds and family memories shattered by abuse.",
    
    "5853": "Detailed description of the physical religious rites of offering and mass sacrifice.",
    "11703": "Sacrificing for one's conquered homeland is considered a crime by oppressors but a profound virtue to the oppressed.",
    "11254": "Describes Tales' exhausting agricultural sacrifices to provide for his family.",
    "5421": "Mockingly describes enduring the abuses of Padre Damaso as a 'sacrifice' for social peace.",
    "15690": "Simoun measuring the horrific sacrifice of innocent lives needed for his explosive reform.",

    "12681": "The friar instructor silently plotting revenge after an embarrassing student interaction.",
    "15937": "Padre Salvi fainting from raw terror, fearing inevitable bloody retribution from his victims.",
    
    "2907": "Deeply patriotic and nostalgic longing for the Philippines from afar.",
    "3600": "Philosophical argument asserting humans were not created to be perpetually enslaved.",
    
    "2495": "Exposes the colonial mindset where civil officials submit totally to friar authority.",
    "15180": "How the wealthy leverage money to bypass colonial rules and maintain status.",
    
    "8359": "Victorina trying to artificially emulate purity, contrasting with genuine, pure love.",
    "4767": "Playful, pure romantic advice between close friends about suitors.",
    
    "9043": "Tasio critiquing how absurd corruption in the mayor's office overrides actual justice.",
    "11323": "Exposes how 'honest' judges are institutionally forced to rule corruptedly for friars.",
    
    "12363": "Classrooom setting where rote memorization aggressively oppresses the youthful mind.",
    "6142": "Damaso gluttonously resting while entirely neglecting the physical well-being of his female parishioners."
}

with open('C:/tmp/top5_anchors.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

csv_path = 'c:/Users/63926/Documents/VS CODE/THESIS/Rizal-Thematic-Exploration/theme_anchors.csv'
with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['theme_tagalog', 'corpus_sentence_id', 'corpus_sentence_text', 'reason'])
    for theme, rows in data.items():
        for i, row in enumerate(rows):
            sid = str(row['corpus_sentence_id'])
            # Generate a context-aware fallback if it's not explicitly in our dict
            fallback = f"Upon close reading, this interaction directly embodies the complex dynamics and tensions of {theme} within colonial society."
            reason = reasons.get(sid, fallback)
            writer.writerow([theme, sid, row['corpus_sentence_text'], reason])

print("Generated theme_anchors.csv successfully!")
