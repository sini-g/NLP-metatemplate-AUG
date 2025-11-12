using System.ComponentModel.DataAnnotations.Schema;

public class OriginalSentence
{
    [DatabaseGenerated(DatabaseGeneratedOption.None)]
    public int Id { get; set; }
     public required string Output { get; set; }
    public required string Task_type { get; set; }
    public required string Context { get; set; }
    public required string Task_input { get; set; }
    public required string Task_output { get; set; }
    public required string Input { get; set; }
    public string ControlStatus { get; set; } = "None";
    public int TimesProposed { get; set; } = 0;
    public int TimesMarkedAsCorrect { get; set; } = 0;
}