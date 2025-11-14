import json
import random
import spacy
import numpy as np
from typing import List, Dict, Any
import babelnet as bn
from babelnet.language import Language
from sentence_transformers import SentenceTransformer, util
from babelnet.pos import POS


class BabelNetAugmenter:
    """Classe per data augmentation usando la libreria BabelNet"""

    def __init__(self):
        """La configurazione viene letta automaticamente da babelnet_conf.yml"""
        print("BabelNet API inizializzata correttamente")

        self.nlp_tool = spacy.load('it_core_news_sm') #Tool per pos embedding e informazioni morfologiche
        self.emb_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') #Embedding per calcolare cosine similarity
        self._emb_cache = {}
        # Mapping tra POS tag di spaCy  e BabelNet POS
        self.pos_mapping = {
            'NOUN': POS.NOUN,
            'PROPN': POS.NOUN,  # Nome proprio -> NOUN
            'VERB': POS.VERB,
            'ADJ': POS.ADJ,
            'ADV': POS.ADV,
            'AUX': POS.VERB,  # Verbi ausiliari -> VERB
        }

        # Contatori per validazione a posteriori delle sostituzioni effettuate
        self.stats = {
            'total_substitutions': 0,  # Totale sostituzioni completate
            'candidates_evaluated': 0,  # Candidati valutati per similarity
            'candidates_passing_threshold': 0,  # Candidati sopra threshold
            'pos_errors_in_substitutions': 0,  # Errori POS nelle sostituzioni finali
            'morphology_errors_in_substitutions': 0,  # Errori morfologici nelle sostituzioni finali
            'general_errors_in_substitutions': 0  # Errori generici nelle sostituzioni finali
        }

    def embed(self, text):
        if text in self._emb_cache:
            return self._emb_cache[text]

        emb = self.emb_model.encode(text, normalize_embeddings=True)
        self._emb_cache[text] = emb

        return emb

    def analyze_text(self, text:str):
        """
        Analizza il testo con spaCy e restituisce
        una lista di dizionari con token, POS tag e informazioni sul genere, numero e tempo verbale.

        Args:
            text: Il testo completo da analizzare

        Returns:
            Lista di dizionari contenenti: 'text', 'pos', 'lemma', 'is_alpha', etc.
        """

        doc = self.nlp_tool(text)

        tokens_info = []
        for token in doc:
            morph_dict = token.morph.to_dict()
            tokens_info.append({
                'text': token.text,
                'pos': token.pos_,
                'babel_pos': self.pos_mapping.get(token.pos_, POS.NOUN),
                'lemma': token.lemma_,
                'is_alpha': token.is_alpha,
                'is_punct': token.is_punct,
                'is_stop': token.is_stop,
                'is_space': token.is_space,
                'gender': morph_dict.get('Gender', None),
                'number': morph_dict.get('Number', None),
                'person': morph_dict.get('Person', None),
                'tense': morph_dict.get('Tense', None),
                'mood': morph_dict.get('Mood', None)
            })

        return tokens_info

    def validate_pos(self, synonym: str, original_pos: str) -> bool:
        """
        Valida che il sinonimo abbia lo stesso POS della parola originale.
        Usato per validazione a posteriori delle sostituzioni.

        Args:
            original_word: Parola originale
            synonym: Sinonimo scelto
            original_pos: POS tag originale

        Returns:
            True se POS corretto, False altrimenti
        """
        doc = self.nlp_tool(synonym)
        syn_token = doc[0]

        # Per verbi ausiliari e verbi normali, consideriamo equivalenti
        if original_pos in ['VERB', 'AUX'] and syn_token.pos_ in ['VERB', 'AUX']:
            return True

        # Per nomi propri e comuni, consideriamo equivalenti
        if original_pos in ['NOUN', 'PROPN'] and syn_token.pos_ in ['NOUN', 'PROPN']:
            return True

        return syn_token.pos_ == original_pos


    def check_morph(self, synonym: str, original_morph: Dict[str, Any]) -> bool:
        """
           Verifica che il sinonimo abbia le stesse caratteristiche morfologiche
           (genere e numero) della parola originale.

           Args:
               original_word: Parola originale
               synonym: Sinonimo candidato
               original_morph: Dizionario con feature morfologiche originali

           Returns:
               True se il sinonimo ha le stesse feature morfologiche
        """
        doc = self.nlp_tool(synonym)
        syn_token = doc[0]
        syn_morph = syn_token.morph.to_dict()

        # Controlla genere (solo per nomi e aggettivi)
        if original_morph.get('gender') is not None:
            if syn_morph.get('Gender') != original_morph['gender']:
                return False

        # Controlla numero (singolare/plurale)
        if original_morph.get('number') is not None:
            if syn_morph.get('Number') != original_morph['number']:
                return False

        # Per i verbi, controlla persona e tempo
        if original_morph.get('person') is not None:
            if syn_morph.get('Person') != original_morph['person']:
                return False

        if original_morph.get('tense') is not None:
            if syn_morph.get('Tense') != original_morph['tense']:
                return False

        return True

    def pick_best_syn(self, word: str, synonyms: list, context: str = None, min_sim : float = 0.5):
        """
           Sceglie il miglior sinonimo tra una lista di sinonimi in base alla cosine similarity

           Args:
               word: Parola originale
               synonyms: Lista di sinonimi
               context: Contesto della parola originale
               min_sim: valore minimo di cosine similarity per accettare un sinonimo

           Returns:
               Migliore tra i sinonimi con la cosine similarity maggiore di min_sim
        """
        if not synonyms:
            return None

        if context:
            query = context
        else:
            query = word

        q_emb = self.embed(query)
        cand_embeds = [self.embed(s) for s in synonyms]

        sims = util.cos_sim(q_emb, np.vstack(cand_embeds)).cpu().numpy().flatten()

        i = int(np.argmax(sims))
        best, best_sims = synonyms[i], float(sims[i])

        # Traccia quanti candidati vengono valutati
        self.stats['candidates_evaluated'] += len(synonyms)
        if best_sims >= min_sim:
            self.stats['candidates_passing_threshold'] += 1
            return best
        else:
            return word

    def get_synonyms(self, word: str, lang: Language = Language.IT, target_pos=POS.NOUN) -> List[str]:
        """
        Ottiene i sinonimi per una parola usando l'API ufficiale BabelNet.

        Args:
            word: La parola per cui cercare sinonimi
            lang: La lingua della parola (default: italiano)
            target_pos: Part-of speech della parola originale
        Returns:
            Lista di sinonimi trovati
        """
        try:
            # Ottieni i synset per la parola
            synsets = bn.get_synsets(word, from_langs=[lang])

            if not synsets:
                return []

            synonyms = set()

            for synset in list(synsets):
                # Ottieni tutti i sense (lessicalizzazioni) del synset
                for sense in synset.senses(lang):
                    if synset.pos != target_pos:
                        continue
                    # lemma = sense.full_lemma
                    normalized = sense.normalized_lemma

                    # Aggiungi solo se diverso dalla parola originale e non contiene caratteri particolari
                    if normalized and normalized.lower() != word.lower() \
                            and '_' not in normalized and '(' not in normalized and '/' not in normalized:
                        synonyms.add(normalized)
                    else:
                        continue

            return list(synonyms)

        except Exception as e:
            print(f"Errore nel recupero sinonimi per '{word}': {e}")
            return []

    def replace_with_synonyms(self, text: str, replacement_rate: float = 0.3) -> str:
        """
        Sostituisce alcune parole con sinonimi.

        Args:
            text: Il testo da aumentare
            replacement_rate: Rate nel quale cambiare le parole
        Returns:
            Testo aumentato con sinonimi
        """
        augmented_words = []
        text_info = self.analyze_text(text)

        words = []
        word_to_token_idx = []

        for idx, token_info in enumerate(text_info):
            if not token_info['is_space']:
                words.append(token_info['text'])
                word_to_token_idx.append(idx)

        # Parole da non sostituire
        skip_words = {'essere', 'avere', 'fare', 'dire', 'che', 'per', 'con', 'una', 'uno',
                      'il', 'lo', 'la', 'i', 'gli', 'le', 'di', 'dei', 'del', 'dello',
                      'della', 'degli', 'delle', 'nel', 'nella', 'nei', 'negli', 'nelle',
                      'sul', 'sulla', 'sui', 'sugli', 'sulle', 'dal', 'dalla', 'dai',
                      'dagli', 'dalle', 'al', 'alla', 'ai', 'agli', 'alle'}

        for i, word in enumerate(words):
            token_idx = word_to_token_idx[i]
            token_info = text_info[token_idx]

            # Pulisci la parola da punteggiatura
            clean_word = word.strip('.,!?;:()"\'').lower()

            if len(clean_word) < 4 or clean_word[0].isupper() or clean_word in skip_words or token_info['is_punct']\
                    or token_info['is_stop']:
                augmented_words.append(word)
                continue

            if random.random() < replacement_rate:
                target_pos = token_info['babel_pos']
                synonyms = self.get_synonyms(word=clean_word, target_pos=target_pos)
                if synonyms:
                    #Filtra i sinonimi per compatibilità morfologica
                    valid_synonyms = []
                    for syn in synonyms:
                        if self.check_morph(syn, {'gender': token_info['gender'],
                                                  'number': token_info['number'],
                                                  'person': token_info['person'],
                                                  'tense': token_info['tense']}
                                           ):
                            valid_synonyms.append(syn)

                    if i > 0 and i < len(words) - 1:
                        context = words[i - 1] + " " + words[i] + " " + words[i + 1]
                    elif i > 0:
                        context = words[i - 1] + " " + words[i]
                    elif i < len(words) - 1:
                        context = words[i] + " " + words[i + 1]
                    else:
                        context = words[i]

                    synonym = self.pick_best_syn(word, valid_synonyms, context)

                    if synonym and synonym != word:
                        if word[0].isupper():
                            synonym = synonym.capitalize()

                        # VALIDAZIONE A POSTERIORI della sostituzione effettuata
                        self.stats['total_substitutions'] += 1

                        # Verifica POS della sostituzione finale
                        if not self.validate_pos(synonym, token_info['pos']):
                            self.stats['pos_errors_in_substitutions'] += 1
                            self.stats['general_errors_in_substitutions'] += 1
                            flag = True #Flag per capire se ci sono errori in entrambe le validazioni
                        else:
                            flag = False

                        # Verifica morfologia della sostituzione finale
                        if not self.check_morph(synonym,
                                                        {'gender': token_info['gender'],
                                                         'number': token_info['number'],
                                                         'person': token_info['person'],
                                                         'tense': token_info['tense']}):
                            self.stats['morphology_errors_in_substitutions'] += 1
                            self.stats['general_errors_in_substitutions'] += 1
                        else:
                            flag = False

                        if flag:
                            self.stats['general_errors_in_substitutions'] -= 1

                        # Riaggiungi la punteggiatura originale
                        suffix = ''.join([c for c in words[i] if not c.isalnum()])
                        augmented_words.append(synonym + suffix)

                        print(f"  Sostituito: '{word}' -> '{synonym + suffix}'")
                    else:
                        augmented_words.append(word)
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)

        return ' '.join(augmented_words)

    def augment_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea una versione aumentata di un elemento del dataset.

        Args:
            element: Elemento originale del dataset
            version: Numero della versione (1 o 2) per variare il tasso di sostituzione

        Returns:
            Elemento aumentato con sinonimi
        """
        augmented = element.copy()


        print(f"\n  Versione aumentata - ")

        # Tasso di sostituzione diverso per ogni versione
        replacement_rate = 0.3

        # Aumenta il campo 'context'
        if 'context' in augmented and augmented['context']:
            print("  Augmenting 'context'...")
            augmented['context'] = self.replace_with_synonyms(augmented['context'], replacement_rate)

        # Aumenta il campo 'task_input'
        if 'task_input' in augmented and augmented['task_input']:
            print("  Augmenting 'task_input'...")
            augmented['task_input'] = self.replace_with_synonyms(augmented['task_input'], replacement_rate)

        # Aumenta il campo 'task_output'
        if 'task_output' in augmented and augmented['task_output']:
            print("  Augmenting 'task_output'...")
            augmented['task_output'] = self.replace_with_synonyms(augmented['task_output'], replacement_rate)

        # Ricostruisci i campi 'input' e 'output' con i nuovi valori
        if 'input' in augmented:
            augmented['input'] = (
                f"<|tasktype|>\n{augmented['task_type']}\n"
                f"<|context|>\n{augmented['context']}\n<|task|>"
            )

        if 'output' in augmented:
            augmented['output'] = (
                f"{augmented['task_input']}\n<|pipe|>\n{augmented['task_output']}"
            )

        return augmented

    def print_statistics(self):
        """
        Stampa le statistiche di validazione delle sostituzioni effettivamente completate.
        Queste metriche dimostrano la qualità dei filtri applicati.
        """
        print(f"\n{'=' * 70}")
        print("STATISTICHE DI VALIDAZIONE SOSTITUZIONI")
        print(f"{'=' * 70}")

        print(f"\nSostituzioni completate:")
        print(f"  • Totale sostituzioni effettuate:           {self.stats['total_substitutions']}")

        print(f"\nCandidati sinonimi valutati:")
        print(f"  • Totale candidati valutati:                {self.stats['candidates_evaluated']}")
        print(f"  • Candidati sopra threshold similarity:     {self.stats['candidates_passing_threshold']}")

        if self.stats['candidates_evaluated'] > 0:
            acceptance_rate = (self.stats['candidates_passing_threshold'] /
                               self.stats['candidates_evaluated']) * 100
            print(f"  • Acceptance rate (threshold):              {acceptance_rate:.2f}%")

        print(f"\nValidazione sostituzioni effettuate:")
        print(f"  • Errori POS nelle sostituzioni:            {self.stats['pos_errors_in_substitutions']}")
        print(f"  • Errori morfologici nelle sostituzioni:    {self.stats['morphology_errors_in_substitutions']}")

        if self.stats['total_substitutions'] > 0:
            pos_error_rate = (self.stats['pos_errors_in_substitutions'] /
                              self.stats['total_substitutions']) * 100
            morph_error_rate = (self.stats['morphology_errors_in_substitutions'] /
                                self.stats['total_substitutions']) * 100

            print(f"  • POS error rate:                           {pos_error_rate:.2f}%")
            print(f"  • Morphology error rate:                    {morph_error_rate:.2f}%")

            # Calcola tasso di correttezza
            correct_substitutions = (self.stats['total_substitutions'] -
                                     self.stats['general_errors_in_substitutions'])
            correctness_rate = (correct_substitutions / self.stats['total_substitutions']) * 100

            print(f"\n  • Sostituzioni corrette:                    {correct_substitutions}")
            print(f"  • Tasso di correttezza complessivo:         {correctness_rate:.2f}%")

        print(f"\n{'=' * 70}\n")


def main():
    """Funzione principale per l'augmentation del dataset"""

    # Carica il dataset originale
    print("\nCaricamento del dataset...")
    with open('../meta_templates_reduced.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    print(f"Dataset caricato: {len(original_data)} elementi")

    # Inizializza l'augmenter
    print("\nInizializzazione BabelNet API...")
    try:
        augmenter = BabelNetAugmenter()
    except Exception as e:
        print(e)
        return

    # Crea il dataset aumentato
    augmented_data = []

    print(f"\n{'=' * 60}")
    print("Inizio processo di augmentation...")

    for idx, element in enumerate(original_data):
        print(f"\n{'─' * 60}")
        print(f"Elemento {idx + 1}/{len(original_data)}")
        print(f"{'─' * 60}")

        # Aggiungi l'elemento originale
        augmented_data.append(element)

        print(f"\n→ Creando versione aumentata...")
        augmented_element = augmenter.augment_element(element)
        augmented_data.append(augmented_element)
        print(f"Versione aumentata completata")


    # Stampa statistiche di validazione
    augmenter.print_statistics()

    # Salva il dataset aumentato
    output_file = '../augmented_data/meta_templates_reduced_augmented.json'
    print(f"Salvataggio del dataset in '{output_file}'...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
