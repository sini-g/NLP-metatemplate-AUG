using Microsoft.EntityFrameworkCore;

public class SentenceService
{
    private readonly IDbContextFactory<CaptchaDbContext> _dbContextFactory;

    public SentenceService(IDbContextFactory<CaptchaDbContext> dbContextFactory)
    {
        _dbContextFactory = dbContextFactory;
    }

    public async Task<OriginalSentence?> GetRandomSentenceAsync()
    {
        using var db = await _dbContextFactory.CreateDbContextAsync();
        var count = await db.OriginalSentences.CountAsync();
        if (count == 0) return null;

        var randomIndex = new Random().Next(0, count);
        
       
        return await db.OriginalSentences.Skip(randomIndex).FirstOrDefaultAsync();
    }
    
}