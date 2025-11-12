using System.Text.Json.Serialization;

public class JsonSentence
{
    [JsonPropertyName("id")]
    public int Id { get; set; }
    [JsonPropertyName("task_type")]
    public string Task_type { get; set; } = string.Empty;
    [JsonPropertyName("context")]
    public string Context { get; set; } = string.Empty;
    [JsonPropertyName("task_input")]
    public string Task_input { get; set; } = string.Empty;
    [JsonPropertyName("task_output")]
    public string Task_output { get; set; } = string.Empty;
    [JsonPropertyName("input")]
    public string Input { get; set; } = string.Empty;
    [JsonPropertyName("output")]
    public string Output { get; set; } = string.Empty;

    [JsonPropertyName("control_status")]
    public string ControlStatus { get; set; } = "None";
}