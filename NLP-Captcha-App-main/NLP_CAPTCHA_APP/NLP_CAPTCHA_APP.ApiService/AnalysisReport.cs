public class AnalysisReport
{
    public CalibrationMetrics? Metrics { get; set; }
    // In futuro aggiungeremo qui:
    // public List<EvaluatedSentence> EvaluatedSentences { get; set; }
    // public List<EvaluatedCorrection> EvaluatedCorrections { get; set; }
}

// Contiene le soglie calcolate dai campioni di controllo "Good" e "Bad"
public class CalibrationMetrics
{
    public double GoodSampleApprovalRate { get; set; }
    public double GoodSampleInteractionTimeAvg { get; set; }
    public double BadSampleApprovalRate { get; set; } // Questo è il nostro "Noise Rate" o "Tasso di Pigrizia"
    public double BadSampleInteractionTimeAvg { get; set; }
    public int TotalGoodSamplesAnalyzed { get; set; }
    public int TotalBadSamplesAnalyzed { get; set; }
}