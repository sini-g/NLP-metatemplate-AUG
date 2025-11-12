using Microsoft.EntityFrameworkCore;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

//Configurazione servizi
builder.Services.AddDbContextFactory<CaptchaDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));
builder.Services.AddScoped<SentenceService>();
builder.Services.AddScoped<StatisticsService>(); 

var app = builder.Build();

app.Use(async (context, next) =>
{
    context.Request.EnableBuffering();
    await next();
});


// Endpoint per il CAPTCHA
app.MapGet("/getsentence", async (CaptchaDbContext db) =>
{
    var count = await db.CaptchaSentences.CountAsync();
    if (count == 0) return Results.NotFound("Nessuna frase per il CAPTCHA disponibile.");
    var randomIndex = new Random().Next(0, count);
    var captchaSentence = await db.CaptchaSentences.Skip(randomIndex).FirstOrDefaultAsync();
    if (captchaSentence is null) return Results.NotFound();
    
    return Results.Ok(new 
    { 
        captchaSentenceId = captchaSentence.Id, 
        originalSentenceId = captchaSentence.OriginalSentenceId, 
        text = captchaSentence.Text 
    });
});

// Endpoint per inviare un risultato al DB 
app.MapPost("/submit-result", async (HttpContext context, CaptchaDbContext db) =>
{
    context.Request.EnableBuffering(); 
    var baseSubmission = await context.Request.ReadFromJsonAsync<SubmissionDto>();
    context.Request.Body.Position = 0;

    if (baseSubmission is null) return Results.BadRequest("Dati mancanti.");

    var originalSentence = await db.OriginalSentences.FindAsync(baseSubmission.SentenceId);
    if (originalSentence is null) return Results.NotFound("Frase originale non trovata.");

    originalSentence.TimesProposed++;

    if (baseSubmission.Action == "MarkedAsCorrect")
    {
        originalSentence.TimesMarkedAsCorrect++;

        
        db.SubmissionLogs.Add(new SubmissionLog
        {
            OriginalSentenceId = baseSubmission.SentenceId,
            InteractionTimeInSeconds = baseSubmission.InteractionTimeInSeconds,
            WasMarkedAsCorrect = true
        });
    }
    else if (baseSubmission.Action == "CorrectionSubmitted")
    {
        var correctionData = await context.Request.ReadFromJsonAsync<CorrectionDto>();
        if (correctionData is null) return Results.BadRequest("Dati correzione mancanti.");

        // Aggiorna la tabella aggregata
        var existingCorrection = await db.SuggestedCorrections.FirstOrDefaultAsync(c =>
            c.OriginalSentenceId == correctionData.SentenceId && c.WordPosition == correctionData.WordPosition &&
            c.OriginalWord == correctionData.OriginalWord && c.SuggestedWord == correctionData.NewWord);

        if (existingCorrection != null) { existingCorrection.SubmissionCount++; }
        else
        {
            db.SuggestedCorrections.Add(new SuggestedCorrection
            {
                OriginalSentenceId = correctionData.SentenceId, WordPosition = correctionData.WordPosition,
                OriginalWord = correctionData.OriginalWord, SuggestedWord = correctionData.NewWord, SubmissionCount = 1
            });
        }

        db.SubmissionLogs.Add(new SubmissionLog
        {
            OriginalSentenceId = correctionData.SentenceId,
            InteractionTimeInSeconds = correctionData.InteractionTimeInSeconds,
            WasMarkedAsCorrect = false,
            WordPosition = correctionData.WordPosition,
            OriginalWord = correctionData.OriginalWord,
            SuggestedWord = correctionData.NewWord
        });
    }

    await db.SaveChangesAsync();
    return Results.Ok("Risultato salvato con successo!");
});

// Endpoint per le Statistiche 
app.MapGet("/statistics/all-sentences", async (CaptchaDbContext db) =>
{
    var sentences = await db.OriginalSentences
        .Select(s => new { s.Id, s.Context }) 
        .OrderBy(s => s.Id)
        .ToListAsync();
    return Results.Ok(sentences);
});

// ENndpoint per i dettagli completi di una frase specifica
app.MapGet("/statistics/sentence-details/{id}", async (int id, CaptchaDbContext db) =>
{
    var sentence = await db.OriginalSentences.FindAsync(id);

    if (sentence is null)
    {
        return Results.NotFound("Frase non trovata.");
    }
    return Results.Ok(sentence);
});

// Endpoint per le correzioni data una frase specifica
app.MapGet("/statistics/corrections/{sentenceId}", async (int sentenceId, CaptchaDbContext db) =>
{
    var corrections = await db.SuggestedCorrections
        .Where(c => c.OriginalSentenceId == sentenceId)
        .OrderByDescending(c => c.SubmissionCount)
        .ToListAsync();
        
    return Results.Ok(corrections);
});

// Endpoint per l'Analisi
app.MapGet("/admin/run-analysis", async (StatisticsService statsService) =>
{
    var report = await statsService.AnalyzeStatisticsAsync();
    return Results.Ok(report);
});


using (var scope = app.Services.CreateScope())
{
    var dbContextFactory = scope.ServiceProvider.GetRequiredService<IDbContextFactory<CaptchaDbContext>>();
    using var dbContext = dbContextFactory.CreateDbContext();
    
    dbContext.Database.EnsureCreated();

    if (!dbContext.OriginalSentences.Any())
    {
        try
        {
            var jsonText = File.ReadAllText("Data/sentences.json");
            var jsonSentences = JsonSerializer.Deserialize<List<JsonSentence>>(jsonText);

            if (jsonSentences != null && jsonSentences.Count > 0)
            {
                var sentencesForDb = jsonSentences.Select(js => new OriginalSentence
                {
                    Id = js.Id,
                    Task_type = js.Task_type,
                    Context = js.Context,
                    Task_input = js.Task_input,
                    Task_output = js.Task_output,
                    Input = js.Input,
                    Output = js.Output,
                    ControlStatus = js.ControlStatus
                }).ToList();

                dbContext.OriginalSentences.AddRange(sentencesForDb);
                dbContext.SaveChanges();
                Console.WriteLine($"SUCCESS: Database popolato con {sentencesForDb.Count} metatemplate, incluso ControlStatus.");
            }
        }
        catch (Exception ex) { Console.WriteLine($"ERRORE durante il popolamento: {ex.Message}"); }
    }
    else { Console.WriteLine("Tabella 'OriginalSentences' già popolata."); }
    await DataPreprocessor.PreprocessSentences(dbContextFactory);
}

app.Run();