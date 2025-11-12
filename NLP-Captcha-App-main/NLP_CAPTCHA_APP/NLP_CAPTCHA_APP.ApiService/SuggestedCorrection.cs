public class SuggestedCorrection
{
    public int Id { get; set; }
    public int OriginalSentenceId { get; set; } 
    public int WordPosition { get; set; } 
    public required string OriginalWord { get; set; }
    public required string SuggestedWord { get; set; }
    public int SubmissionCount { get; set; } = 1;
}