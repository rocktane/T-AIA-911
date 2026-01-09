# VIA - Voice Improvement Approach

Plan d'amelioration de la reconnaissance vocale pour le domaine ferroviaire francais.

## Probleme

La reconnaissance vocale Whisper transcrit les homophones sans contexte de domaine :
- "l'ile" au lieu de "Lille"
- "l'ion" au lieu de "Lyon"
- "Nice" vs "nice" (adjectif)
- "Tours" vs "tours" (nom commun)
- "Lourdes" vs "l'ourde"
- "Rennes" vs "reine"

## Solution : Approche Hybride

Combiner deux mecanismes complementaires pour une couverture maximale.

### 1. Prompt Whisper (biasing de domaine)

**Fichier a modifier** : `src/web/templates/index.html` (ligne ~1159)

**Principe** : Whisper accepte un parametre `prompt` qui biaise la transcription vers un vocabulaire/contexte specifique.

**Implementation** :
```javascript
const result = await voiceState.transcriber(audioData, {
    language: 'french',
    task: 'transcribe',
    prompt: `Voyage en train SNCF. Villes et gares: Paris, Lyon, Marseille, Lille,
Bordeaux, Toulouse, Nice, Nantes, Strasbourg, Montpellier, Rennes, Le Havre,
Reims, Saint-Etienne, Toulon, Grenoble, Dijon, Angers, Nimes, Tours,
Saint-Denis, Avignon, Clermont-Ferrand, Le Mans, Aix-en-Provence, Brest,
Limoges, Amiens, Perpignan, Besancon, Orleans, Rouen, Metz, Mulhouse, Caen.
Je veux aller de Paris a Lyon. Depart Lille arrivee Marseille.`
});
```

**Avantages** :
- Biaise vers le domaine voyage/train
- Ameliore les villes courantes
- Pas de latence supplementaire

**Limites** :
- ~224 tokens max pour le prompt
- Ne couvre pas les 2785 gares

### 2. Post-processing Phonetique

**Nouveau fichier** : `src/web/static/js/phonetic-correction.js`

**Principe** : Apres transcription, analyser le texte et corriger les segments qui ressemblent phonetiquement a des villes connues.

**Implementation** :

```javascript
// Dictionnaire d'homophones connus (cas frequents)
const HOMOPHONES = {
    "l'ile": "Lille",
    "l'île": "Lille",
    "lil": "Lille",
    "l'ion": "Lyon",
    "lion": "Lyon",
    "l'yon": "Lyon",
    "l'ourde": "Lourdes",
    "reine": "Rennes",
    "la reine": "Rennes",
    "tour": "Tours",
    "la tour": "Tours",
    "nice": "Nice",  // Contexte voyage
    "mans": "Le Mans",
    "le man": "Le Mans",
    "angers": "Angers",  // vs "en jair"
    // ... etc
};

// Fonction de correction phonetique francaise (Soundex FR ou phonetisation)
function frenchPhonetic(word) {
    // Normalisation
    let s = word.toLowerCase()
        .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
        .replace(/[^a-z]/g, '');

    // Regles phonetiques francaises
    s = s.replace(/ph/g, 'f')
         .replace(/qu/g, 'k')
         .replace(/gu/g, 'g')
         .replace(/ch/g, 'x')
         .replace(/eau/g, 'o')
         .replace(/au/g, 'o')
         .replace(/ou/g, 'u')
         .replace(/ai/g, 'e')
         .replace(/ei/g, 'e')
         .replace(/oi/g, 'wa')
         .replace(/an|en|am|em/g, 'a')
         .replace(/on|om/g, 'o')
         .replace(/in|im|ain|ein|un/g, 'e')
         .replace(/[eiy]$/g, 'i')
         .replace(/[dt]$/g, '')
         .replace(/s$/g, '')
         .replace(/(.)\1+/g, '$1');  // Dedupe

    return s;
}

// Construire l'index phonetique des villes (au chargement)
function buildPhoneticIndex(cities) {
    const index = new Map();
    for (const city of cities) {
        const phonetic = frenchPhonetic(city);
        if (!index.has(phonetic)) {
            index.set(phonetic, []);
        }
        index.get(phonetic).push(city);
    }
    return index;
}

// Corriger la transcription
function correctTranscription(text, phoneticIndex) {
    // 1. Corrections directes (homophones connus)
    let corrected = text;
    for (const [wrong, right] of Object.entries(HOMOPHONES)) {
        const regex = new RegExp(`\\b${wrong}\\b`, 'gi');
        corrected = corrected.replace(regex, right);
    }

    // 2. Correction phonetique pour mots non reconnus
    const words = corrected.split(/\s+/);
    const result = [];

    for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const phonetic = frenchPhonetic(word);

        // Si le mot correspond phonetiquement a une ville
        if (phoneticIndex.has(phonetic)) {
            const candidates = phoneticIndex.get(phonetic);
            // Prendre la ville la plus probable (ou la premiere)
            result.push(candidates[0]);
        } else {
            result.push(word);
        }
    }

    return result.join(' ');
}
```

### 3. Integration dans le flux

**Fichier** : `src/web/templates/index.html`

```javascript
// Au chargement de la page, construire l'index phonetique
let phoneticIndex = null;

async function loadCitiesForPhonetic() {
    const response = await fetch('/api/cities');  // Nouvel endpoint
    const cities = await response.json();
    phoneticIndex = buildPhoneticIndex(cities);
}

// Dans transcribeAudio()
async function transcribeAudio() {
    setVoiceStatus('transcribing');

    try {
        const audioBlob = new Blob(voiceState.audioChunks, { type: 'audio/webm' });
        const audioData = await processAudioBlob(audioBlob);

        // Transcription avec prompt de domaine
        const result = await voiceState.transcriber(audioData, {
            language: 'french',
            task: 'transcribe',
            prompt: TRAVEL_PROMPT  // Nouveau: prompt de domaine
        });

        // Post-processing phonetique
        let text = result.text.trim();
        if (phoneticIndex) {
            text = correctTranscription(text, phoneticIndex);
        }

        if (text) {
            sentenceInput.value = text;
            updateClearButton();
            startNewSearch();
        }
    } catch (error) {
        console.error('Erreur transcription:', error);
    }
}
```

### 4. Nouvel endpoint API

**Fichier** : `src/web/app.py`

```python
@app.get("/api/cities")
async def get_cities():
    """Return list of all city/station names for phonetic matching."""
    cities = set()

    # Ajouter les noms de gares
    if city_resolver and city_resolver.station_finder:
        for station in city_resolver.station_finder.stations:
            cities.add(station.name)
            # Extraire aussi le nom de ville (avant le premier espace/tiret)
            base_name = station.name.split()[0].split('-')[0]
            if len(base_name) > 2:
                cities.add(base_name)

    # Ajouter les communes
    if city_resolver:
        for city in city_resolver.cities.values():
            cities.add(city.name)

    return sorted(cities)
```

## Fichiers a modifier

| Fichier | Modification |
|---------|-------------|
| `src/web/templates/index.html` | Ajouter prompt Whisper + correction phonetique |
| `src/web/app.py` | Ajouter endpoint `/api/cities` |

## Etapes d'implementation

1. **Ajouter le prompt Whisper** dans `transcribeAudio()` - impact immediat sur les villes courantes

2. **Creer le dictionnaire d'homophones** - corrections directes pour les cas frequents

3. **Implementer la phonetisation francaise** - algorithme de comparaison phonetique

4. **Ajouter l'endpoint `/api/cities`** - exposer la liste des villes au frontend

5. **Integrer le post-processing** - corriger apres transcription

6. **Tester** avec des phrases problematiques :
   - "Je veux aller a Lille" (vs "l'ile")
   - "De Lyon a Rennes" (vs "lion a reine")
   - "Depart Tours arrivee Nice" (vs "tour nice")

## Metriques de succes

- "l'ile" → "Lille" ✓
- "l'ion" → "Lyon" ✓
- "reine" → "Rennes" ✓ (dans contexte voyage)
- "la tour" → "Tours" ✓
- Villes rares reconnues via phonetique

## Notes techniques

- Le prompt Whisper est limite a ~224 tokens
- L'index phonetique est construit une seule fois au chargement
- La correction est instantanee (O(n) sur les mots)
- Compatible avec le fuzzy matching existant de spaCy/CamemBERT
