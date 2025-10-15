from sentence_transformers import SentenceTransformer, util

def evaluate_semantic_similarity(original_texts, augmented_texts):
    """Valuta la similarità semantica tra testi originali e aumentati"""

    # 1. Sentence-BERT
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    results = []
    emb_orig = model.encode(original_texts, convert_to_tensor=True)
    emb_aug = model.encode(augmented_texts, convert_to_tensor=True)

    # Cosine similarity
    cosine_sim = util.cos_sim(emb_orig, emb_aug).item()

    results.append({
        'cosine_similarity': cosine_sim,
    })

    return results

if __name__ == "__main__":
    original = "Su una barca nel Mare del Nord, tre uomini stanno importando droga nell'Essex: Mickey Steele, Darren Nicholls e Jack Whomes. All'insaputa degli altri due, Nicholls è un informatore della polizia che ha detto a D.I. Stone, un agente di polizia, della droga. Le droghe, tuttavia, raggiungono ancora Essex perché Steele anticipa i problemi e manda via Whomes su una barca con il contrabbando.\nViene rivelato da Nicholls, che è il narratore del film, che i tre uomini sono fornitori di uno spacciatore dell'Essex di nome Tony Tucker. Tucker, il suo braccio destro Craig Rolfe e lo psicotico Patrick  Pat Tate sono al servizio come i tre membri principali dei ragazzi dell'Essex. La banda cresce progressivamente in spessore fino a quando una ragazza finisce in coma e poi muore dopo aver preso una pillola di ecstasy pura.\nEnraged, Tucker e Tate visitano Steele e lo minacciano. Per ripagarli, Steele racconta loro di un lavoro ad Amsterdam, che Nicholls, Tate, Rolfe e Steele completano con successo. Nicholls, tuttavia, è devastato dal senso di colpa dopo aver ucciso tre uomini. Nel frattempo, Tate si sente come inarrestabile e tradisce la sua partner Karen, solo per farsi lasciare da lei per Steele. Inoltre aggredisce brutalmente un impiegato della pizzeria perché il dipendente si rifiuta di fare una pizza su misura per il nuovo partner di Tate, dando alla polizia accuse solide verso un membro per la prima volta. Nonostante ciò, Stone dice al dipendente di ritirare le accuse perché sa che è necessaria una condanna a lungo termine. Tuttavia, per questa decisione è sotto il controllo dei suoi superiori."
    augmented =  "Su una imbarcazione nel Mare del Norte , tre uomini stanno importando droga nell' Essex : Mickey Steele , Darren Nicholls e Jack-2 Whomes . All' insaputa degli altri due , Nicholls è un informatore della polizia che ha detto a D.I. Stone , un agente di polizia , della farmacognosia . Le droghe , tuttavia , raggiungono ancora Essex perché Steele anticipa i problemi e manda via Whomes su una imbarcazione con il contrabbando . Viene rivelato da Nicholls , che è il narratore del film , che i tre maschi sono fornitori di uno spacciatore dell' Essex di nome Tony Tucker . Tucker , il suo braccio destro Craig Rolfe e lo psicotico Patrick  Pat Tate sono al servizio come i tre membri principali dei ragazzi dell' Essex . La banda cresce progressivamente in spessore fino a quando una femminuccia finisce in comatoseness e poi muore dopo aver preso una pastiglia di metilendiossimetanfetamina  pura  . Enraged , Tucker e Tate visitano Steele e lo minacciano . Per ripagarli , Steele racconta loro di un lavoro ad Amsterdam , che Nicholls , Tate , Rolfe e Steele completano con successo . Nicholls , tuttavia , è devastato dal senso di colpa dopo aver ucciso tre uomini . Nel frattempo , Tate si sente come  irreversibile  e tradisce la sua partner Karen , solo per farsi partire da lei per Steele . Inoltre aggredisce brutalmente un impiegato della pizzeria perché il dipendente si rifiuta di fare una pizza su taglia per il nuovo compagno di Tate , dando alla gendarmeria accuse solide verso un associato per la prima volta . Nonostante ciò , Stone dice al impiegato di ritirare le accuse perché sa che è necessaria una condanna a lungo termine . Tuttavia , per questa decisione è sotto il autocontrollo dei suoi superiori ."

    print(evaluate_semantic_similarity(original, augmented))