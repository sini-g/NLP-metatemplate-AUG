public class SubmissionLog
{
    public int Id { get; set; }
    public int OriginalSentenceId { get; set; } 
    public DateTime Timestamp { get; set; } = DateTime.UtcNow; 
    public double InteractionTimeInSeconds { get; set; } 
    public bool WasMarkedAsCorrect { get; set; } 

    // campi correzione
    public int? WordPosition { get; set; }
    public string? OriginalWord { get; set; }
    public string? SuggestedWord { get; set; }
}