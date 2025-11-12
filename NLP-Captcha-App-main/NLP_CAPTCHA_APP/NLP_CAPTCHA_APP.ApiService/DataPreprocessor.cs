using Microsoft.EntityFrameworkCore;
using System.Text.RegularExpressions;

public static class DataPreprocessor
{
    private const int MaxWordsPerChunk = 15; 
    private const int MinWordsPerChunk = 5; 

    public static async Task PreprocessSentences(IDbContextFactory<CaptchaDbContext> dbFactory)
    {
        using var db = await dbFactory.CreateDbContextAsync();
        if (await db.CaptchaSentences.AnyAsync())
        {
            Console.WriteLine("Le frasi per il CAPTCHA sono già state pre-processate.");
            return;
        }

        Console.WriteLine("Avvio del preprocessing intelligente delle frasi...");
        var originalSentences = await db.OriginalSentences.ToListAsync();
        var captchaSentencesToAdd = new List<CaptchaSentence>();

        foreach (var original in originalSentences)
        {
            var tokens = Tokenize(original.Context);
            var segments = SegmentBySentence(tokens);
            int totalOriginalWordOffset = 0;

            foreach (var segment in segments)
            {
                var segmentClickableTokens = segment.Where(t => t.IsClickable).ToList();
                int segmentWordCount = segmentClickableTokens.Count;

                if (segmentWordCount == 0) continue;

                int currentOffsetInSegment = 0;
                while (currentOffsetInSegment < segmentWordCount)
                {
                    var chunkTokens = GetTokensForChunk(segment, currentOffsetInSegment, MaxWordsPerChunk);
                    int chunkWordCount = chunkTokens.Count(t => t.IsClickable);

                    if (chunkWordCount == 0) break;


                    int remainingWords = segmentWordCount - (currentOffsetInSegment + chunkWordCount);
                    if (remainingWords > 0 && remainingWords < MinWordsPerChunk)
                    {
                        var remainingTokens = GetTokensForChunk(segment, currentOffsetInSegment + chunkWordCount, remainingWords);
                        chunkTokens.AddRange(remainingTokens);
                        chunkWordCount += remainingWords;
                    }
                  

                    captchaSentencesToAdd.Add(new CaptchaSentence
                    {
                        OriginalSentenceId = original.Id,
                        Text = ReconstructText(chunkTokens),
                        WordOffset = totalOriginalWordOffset + currentOffsetInSegment
                    });

                    currentOffsetInSegment += chunkWordCount;
                }
                totalOriginalWordOffset += segmentWordCount;
            }
        }

        await db.CaptchaSentences.AddRangeAsync(captchaSentencesToAdd);
        await db.SaveChangesAsync();
        Console.WriteLine($"SUCCESS: Preprocessing intelligente completato. Generate {captchaSentencesToAdd.Count} frasi per il CAPTCHA.");
    }


    private class Token
    {
        public string Text { get; set; } = "";
        public bool IsClickable { get; set; }
    }

    private static List<Token> Tokenize(string text)
    {
        var regex = new Regex(@"\w+|[^\s\w]");
        return regex.Matches(text)
            .Select(match => new Token
            {
                Text = match.Value,
                IsClickable = Regex.IsMatch(match.Value, @"\w")
            })
            .ToList();
    }

    private static string ReconstructText(List<Token> tokens)
    {
        return string.Join(" ", tokens.Select(t => t.Text))
                     .Replace(" .", ".").Replace(" ,", ",").Replace(" ?", "?").Replace(" !", "!");
    }

    private static List<List<Token>> SegmentBySentence(List<Token> tokens)
    {
        var segments = new List<List<Token>>();
        var currentSegment = new List<Token>();
        foreach (var token in tokens)
        {
            currentSegment.Add(token);
            if (token.Text == ".")
            {
                segments.Add(currentSegment);
                currentSegment = new List<Token>();
            }
        }
        if (currentSegment.Any()) segments.Add(currentSegment); 
        return segments;
    }

    private static List<Token> GetTokensForChunk(List<Token> segmentTokens, int startWordIndex, int maxWords)
    {
        var chunk = new List<Token>();
        int wordCount = 0;
        int currentIndex = 0;
        
        for (int i = 0; i < segmentTokens.Count; i++)
        {
            if (segmentTokens[i].IsClickable)
            {
                if (currentIndex >= startWordIndex)
                {
                    currentIndex = i;
                    break;
                }
                currentIndex++;
            }
        }

        for (int i = currentIndex; i < segmentTokens.Count; i++)
        {
            chunk.Add(segmentTokens[i]);
            if (segmentTokens[i].IsClickable)
            {
                wordCount++;
                if (wordCount >= maxWords) break;
            }
        }
        return chunk;
    }
}