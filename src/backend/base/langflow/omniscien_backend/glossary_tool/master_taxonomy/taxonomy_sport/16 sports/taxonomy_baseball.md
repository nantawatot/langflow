# Baseball Sports Taxonomy
```json
{
    "level 1": {
        "category": "Baseball Sports",
        "description": "a bat and ball game in which the aim is to hit the ball and score runs by running around four bases."
    },
    "level 2": {
        "category": "Standard Baseball",
        "description": "There are two teams, nine players in each, and they take turns in batting and fielding. Batting is done by the offense team and the fielding is played by the defense team.",
        "reference_website": "https://www.topendsports.com/sport/list/baseball.htm"
    }
}
```
### Core Attributes:
- **Sport Category**: Baseball Sports, The general category of the sport that includes all types of baseball competitions and formats.
- **Sport**: Standard Baseball, The most recognized form of baseball, including professional leagues and organized competitions.
- **League Classification**: Classification of competitive level, e.g., Major League, Minor League.
- **League**: Subdivisions or conferences within a league, e.g., National League, American League.
- **Competition/Event**: Names of tournaments, e.g., World Series, All-Star Game, MLB Playoffs.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Round/Game**: Competition stages, e.g., Wild Card Game, Division Series, Championship Series.
- **Team/Club**: Participating teams, e.g., New York Yankees, Los Angeles Dodgers, Boston Red Sox.
  - **Athlete**: Player names, e.g., Aaron Judge, Mookie Betts, Shohei Ohtani.
    - Role/Position: Player roles on the team, e.g., Pitcher, Catcher, Shortstop, Outfielder, Designated Hitter.
    - Equipment/Brand/Model: Player gear, e.g., Louisville Slugger baseball bat, Rawlings glove.
  - **Management**: Team staff, e.g., Manager, General Manager, Coach.
  - **Sponsors**: Team sponsors, e.g., Nike, Under Armour, Gatorade.
  - **Fanbase**: Team supporter groups, e.g., "Yankees Nation," "Red Sox Nation."

- **Venue**: Competition locations, e.g., Yankee Stadium, Fenway Park, Dodger Stadium.
- **Ranking**: Team rankings, e.g., MLB Power Rankings, Wild Card Standings.
- **League**: Related leagues, e.g., National League, American League, Pacific Coast League.
- **Award**: Honors received, e.g., MVP (Most Valuable Player), Cy Young Award, Rookie of the Year.
- **Broadcast**: Media outlets broadcasting games, e.g., ESPN, Fox Sports, MLB Network.


### Example Taxonomy Structure for Baseball:
```plaintext
Baseball Sports → Sport Category
A bat-and-ball sport played between two teams aiming to score runs by hitting a pitched ball and running bases. Baseball has multiple leagues and competitive levels worldwide.

└── Standard Baseball → The most recognized form of baseball, including professional leagues and organized competitions.

    └── LeagueClassification → Classification of competitive level.
        ├── Major League → Top-tier professional competition (e.g., Major League Baseball - MLB).
        └── Minor League → Affiliate or developmental leagues.

    └── League → Organizational subdivisions within the league.
        ├── Name → e.g., National League, American League.

        └── Competition/Event → Organized tournaments or series.
            ├── Name → e.g., World Series, All-Star Game, Playoffs.
            ├── Season/Year → Calendar year of the competition (e.g., 2025 Season).

            └── Venue → Location of the event.
                ├── Stadium → Name of stadium/ballpark (e.g., Yankee Stadium, Dodger Stadium).
                └── Location → City or geographic region.

            └── Round/Game → Phases of the competition.
                ├── GameType → e.g., Regular Season, Wild Card, Division Series, Championship Series, World Series.
                └── MatchNumber → Specific game identifier (e.g., Game 1, Game 5).

            └── Team/Club → Participating teams.
                ├── TeamName → e.g., New York Yankees, Los Angeles Dodgers.
                ├── Management → Staff leadership.
                    ├── Manager → Head coach of the team.
                    └── General Manager → Oversees team operations and roster moves.
                ├── Sponsors → Brands supporting the team (e.g., Nike, Gatorade).
                ├── Ranking → Standings or league position (e.g., 1st in AL East).
                ├── Fanbase → Nickname or identity of team supporters (e.g., "Red Sox Nation").

                └── Athletes → Active roster of players.
                    ├── AthleteName → Full name of player.
                    ├── Role/Position → Field role (e.g., Pitcher, Catcher, Shortstop).
                    ├── Time/Record → Individual stats (e.g., ERA, Home Runs).

                    └── Equipment → Player-specific gear.
                        ├── Bat → Model and brand (e.g., Louisville Slugger).
                        ├── Glove → Model and brand (e.g., Rawlings).
                        └── Accessories → e.g., helmet, cleats, wristbands.

    └── Award → Honors and seasonal recognitions.
        ├── AwardName → e.g., Most Valuable Player (MVP), Cy Young Award, Rookie of the Year.
        └── Recipient → Name of player or team awarded.

    └── Broadcast → Media platforms distributing content.
        ├── Network → e.g., ESPN, Fox Sports, MLB.tv.
        └── CoverageType → e.g., Live, Replay, Highlights, On-demand.

```