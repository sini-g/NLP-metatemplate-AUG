using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace NLP_CAPTCHA_APP.ApiService.Migrations
{
    /// <inheritdoc />
    public partial class FinalSchema : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "OriginalSentences",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false),
                    Output = table.Column<string>(type: "TEXT", nullable: false),
                    Task_type = table.Column<string>(type: "TEXT", nullable: false),
                    Context = table.Column<string>(type: "TEXT", nullable: false),
                    Task_input = table.Column<string>(type: "TEXT", nullable: false),
                    Task_output = table.Column<string>(type: "TEXT", nullable: false),
                    Input = table.Column<string>(type: "TEXT", nullable: false),
                    ControlStatus = table.Column<string>(type: "TEXT", nullable: false),
                    TimesProposed = table.Column<int>(type: "INTEGER", nullable: false),
                    TimesMarkedAsCorrect = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_OriginalSentences", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "SubmissionLogs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    OriginalSentenceId = table.Column<int>(type: "INTEGER", nullable: false),
                    Timestamp = table.Column<DateTime>(type: "TEXT", nullable: false),
                    InteractionTimeInSeconds = table.Column<double>(type: "REAL", nullable: false),
                    WasMarkedAsCorrect = table.Column<bool>(type: "INTEGER", nullable: false),
                    WordPosition = table.Column<int>(type: "INTEGER", nullable: true),
                    OriginalWord = table.Column<string>(type: "TEXT", nullable: true),
                    SuggestedWord = table.Column<string>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_SubmissionLogs", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "SuggestedCorrections",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    OriginalSentenceId = table.Column<int>(type: "INTEGER", nullable: false),
                    WordPosition = table.Column<int>(type: "INTEGER", nullable: false),
                    OriginalWord = table.Column<string>(type: "TEXT", nullable: false),
                    SuggestedWord = table.Column<string>(type: "TEXT", nullable: false),
                    SubmissionCount = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_SuggestedCorrections", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "CaptchaSentences",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    OriginalSentenceId = table.Column<int>(type: "INTEGER", nullable: false),
                    Text = table.Column<string>(type: "TEXT", nullable: false),
                    WordOffset = table.Column<int>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_CaptchaSentences", x => x.Id);
                    table.ForeignKey(
                        name: "FK_CaptchaSentences_OriginalSentences_OriginalSentenceId",
                        column: x => x.OriginalSentenceId,
                        principalTable: "OriginalSentences",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_CaptchaSentences_OriginalSentenceId",
                table: "CaptchaSentences",
                column: "OriginalSentenceId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "CaptchaSentences");

            migrationBuilder.DropTable(
                name: "SubmissionLogs");

            migrationBuilder.DropTable(
                name: "SuggestedCorrections");

            migrationBuilder.DropTable(
                name: "OriginalSentences");
        }
    }
}
