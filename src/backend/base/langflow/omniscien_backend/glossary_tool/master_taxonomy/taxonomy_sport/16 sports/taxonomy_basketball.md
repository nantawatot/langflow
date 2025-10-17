# Basketball Sports Taxonomy
```json
{
    "level 1": {
        "category": "Basketball Sports",
        "description": "players attempt to shoot the ball through the hoop on the opponent\u2019s court, moving the ball by throwing and dribbling."
    },
    "level 2": {
        "category": "Basketball",
        "description": "players attempt to shoot the ball through the hoop on the opponent\u2019s court, moving the ball by throwing and dribbling.",
        "reference_website": "https://www.topendsports.com/sport/list/basketball.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Basketball Sports, The general category of the sport that includes all types of basketball competitions and formats.
- **Sport**: Basketball, A team sport where two teams compete to score points by shooting a ball through the opponent's hoop. It is played professionally and at amateur levels worldwide with various leagues and tournaments.
- **League Classification**: Classification of competitive level, e.g., Professional, Amateur, College.
- **League**: Organized basketball competitions and leagues, such as NBA (National Basketball Association) and NCAA (college basketball).
- **Discipline Type**: Formats or phases within the league, e.g., Regular Season, Playoffs, All-Star Game.
- **Competition/Event**: Named basketball events or championships, e.g., NBA Finals, March Madness (NCAA Tournament), FIBA World Cup.
- **Season/Year**: The year or season the event takes place (e.g., 2023).
- **Round/Match/Quarter**: Competition stages, e.g., Quarter Finals, Semi-finals, Final.
- **Team/Club**: Basketball teams participating in the event (e.g., Los Angeles Lakers, Boston Celtics, Golden State Warriors).
  - **Athlete**: Players on the team, such as LeBron James, Michael Jordan, Stephen Curry.
    - **Role/Position**: Player positions and responsibilities, e.g., Point Guard, Shooting Guard, Small Forward, Power Forward, Center.
    - **Equipment/Brand/Model**: Gear used by players, e.g., Nike Air Zoom Freak shoes, Wilson Evolution basketball.
  - **Management**: Team staff such as Head Coach, Assistant Coach, General Manager.
  - **Sponsors**: Companies sponsoring teams or players (e.g., Nike, Adidas, Coca-Cola).
  - **Ranking**: Team or player rankings, such as NBA Power Rankings, FIBA World Rankings.
  - **Fanbase**: Fan groups or supporter communities, e.g., "Lakers Nation," "Celtics Pride."
- **Referee**: Officials responsible for enforcing rules during games, e.g., Earl Strom, Joey Crawford.
- **Award**: Honors and recognitions like MVP (Most Valuable Player), Rookie of the Year.
- **Broadcasting**: Media outlets airing the games, such as ESPN, TNT, ABC Sports.
- **Venue**: Competition locations, e.g., Staples Center, Madison Square Garden, TD Garden.


### Example Taxonomy Structure for Basketball:
```plaintext
Basketball Sports → Sport Category
A team sport where two teams compete to score points by shooting a ball through the opponent's hoop. Played at professional and amateur levels globally through various leagues and tournaments.

└── Basketball → The specific basketball format focused on here.

    └── League → Official basketball competitions and organizations.
        ├── Name → e.g., NBA (National Basketball Association), NCAA (College Basketball), FIBA (International Federation).

        └── Discipline Type → Organizational format within the league.
            ├── Regular Season → Scheduled games forming the primary phase of competition.
            ├── Playoffs → Post-season elimination rounds (e.g., Quarterfinals, Semifinals, Finals).
            └── All-Star Game → Showcase of top players in exhibition format.

            └── Competition/Event → Official tournaments or series.
                ├── Name → e.g., NBA Finals, March Madness, FIBA World Cup.
                ├── Season/Year → Calendar year or seasonal cycle (e.g., 2024–25 Season).

                └── Venue → Game location.
                    ├── Arena → e.g., Madison Square Garden, Staples Center, TD Garden.
                    └── City/Country → e.g., Los Angeles, Tokyo, Paris.

                └── Round/Match/Quarter → Competitive phase or in-game segment.
                    ├── Round → e.g., Quarterfinals, Semifinals, Final.
                    └── Match Segment → e.g., Q1, Q2, Q3, Q4 (Quarters), Overtime.

                └── Team/Club → Participating teams.
                    ├── TeamName → e.g., Los Angeles Lakers, Boston Celtics, Golden State Warriors.

                    └── Athlete → Team roster.
                        ├── AthleteName → e.g., LeBron James, Stephen Curry.
                        ├── Role/Position → e.g., Point Guard, Shooting Guard, Small Forward, Power Forward, Center.

                        └── Equipment/Brand/Model → Gear used.
                            ├── Shoes → e.g., Nike Air Zoom Freak, Adidas Harden Vol. 7.
                            └── Jersey/Uniform → Team-issued apparel.

                    ├── Management → Team leadership.
                        ├── Head Coach → e.g., Erik Spoelstra, Gregg Popovich.
                        ├── Assistant Coach
                        └── General Manager

                    ├── Sponsors → Brands affiliated with the team or athletes (e.g., Nike, Adidas, Gatorade).
                    ├── Ranking → Standings or performance metrics (e.g., NBA Power Rankings, FIBA Rankings).
                    └── Fanbase → Community identity of supporters (e.g., "Lakers Nation", "Celtics Pride").

                └── Referee → Officials managing gameplay and rule enforcement.
                    ├── Name → e.g., Joey Crawford, Earl Strom.
                    └── Role → Lead Referee, Assistant Referee, Replay Official.

                └── Award → Official recognitions.
                    ├── MVP → Most Valuable Player.
                    ├── Rookie of the Year
                    ├── Defensive Player of the Year
                    └── Coach of the Year

                └── Broadcasting → Media and distribution.
                    ├── Network → e.g., ESPN, TNT, ABC Sports.
                    └── CoverageType → e.g., Live, Replay, Highlight Show, International Feed.


```