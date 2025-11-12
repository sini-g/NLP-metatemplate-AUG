using Microsoft.AspNetCore.Http.Connections;
using NLP_CAPTCHA_APP.Web.Components;

var builder = WebApplication.CreateBuilder(args);

// ---- MODIFICHE ----
// builder.AddServiceDefaults(); // <-- COMMENTATO!

builder.Services.Configure<HttpConnectionDispatcherOptions>(options =>
{
    options.ApplicationMaxBufferSize = 1024 * 1024;
    options.TransportMaxBufferSize = 1024 * 1024;
});

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddOutputCache();

builder.Services.AddHttpClient(); 
var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseAntiforgery();
app.UseOutputCache();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

// app.MapDefaultEndpoints(); // <-- COMMENTATO!

app.Run();