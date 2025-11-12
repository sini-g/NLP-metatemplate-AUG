public class CaptchaSentence
{
    public int Id { get; set; } 
    
    // Chiave esterna che collega questa sotto-frase al suo metatemplate originale
    public int OriginalSentenceId { get; set; }
    public OriginalSentence OriginalSentence { get; set; } = null!;
    public required string Text { get; set; } 
    public int WordOffset { get; set; }
}