public class SubmissionDto
{
    public int SentenceId { get; set; }
    public string Action { get; set; } = string.Empty;
    public double InteractionTimeInSeconds { get; set; }
}

public class CorrectionDto : SubmissionDto
{
    public int WordPosition { get; set; }
    public string OriginalWord { get; set; } = string.Empty;
    public string NewWord { get; set; } = string.Empty;
}