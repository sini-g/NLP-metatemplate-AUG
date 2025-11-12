var builder = DistributedApplication.CreateBuilder(args);

// Aggiunge il progetto del backend (il nostro servizio API)
var apiservice = builder.AddProject<Projects.NLP_CAPTCHA_APP_ApiService>("apiservice");

// Aggiunge il progetto del frontend (la nostra app Blazor)
// e gli dice che può comunicare con il backend (WithReference)
// E, soprattutto, che deve essere accessibile dall'esterno!
builder.AddProject<Projects.NLP_CAPTCHA_APP_Web>("webfrontend")
       .WithReference(apiservice)
       .WithExternalHttpEndpoints(); // <-- QUESTA È LA RIGA FONDAMENTALE DA AGGIUNGERE

builder.Build().Run();

