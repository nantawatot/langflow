# Badminton Taxonomy

```json{
    "level 1": {
        "category": "Racket Sports",
        "description": "such as tennis, squash, badminton and table tennis."
    },
    "level 2": {
        "category": "Badminton",
        "description": "is a type of racket sport that is either played indoors by individuals (singles) or a team of two (doubles)",
        "reference_website": "https://www.topendsports.com/sport/list/badminton.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Racket Sports
- **Sport**: Badminton
- **Competition/Event**: Names of tournaments, e.g., All England Open, BWF World Championships, Thomas Cup.
- **Discipline Type**: Men's Singles, Women's Singles, Men's Doubles, Women's Doubles, Mixed Doubles and others.
- **Season/Year**: Year the event takes place, e.g., 2023.
- **DivisionCategoryBracket**:
  - **Main Draw**: The main competition bracket for the tournament.
    - **MatchRound**: Stages of the tournament, e.g., Round of 64, Round of 32, Quarterfinals, Semifinals, Final.
- **Athlete**: player attend to the tournament, e.g., Kento Momota, Tai Tzu-ying, Viktor Axelsen, P.V. Sindhu.
  - **PlayerName**: Full name of the player.
  - **Country**: Country represented by the player.
  - **WorldRanking**: BWF World Ranking.
  - **Coach**: Name of the coach.
  - **TitlesHeld**: Relevant titles or championships won.
- **Referee**: Name or title of match referee.
- **Management**: Team officials, e.g., Coach, National Team Manager, Technical Director.
- **Sponsors**: Player sponsors, e.g., Yonex, Li-Ning, Victor.
- **Ranking**: Player rankings, e.g., BWF World Ranking.
- **Venue**: Tournament locations, e.g., Emirates Arena, Arena Birmingham, Indoor Stadium.
- **Prize Money/Rewards**: Awards and prize amounts, e.g., $50,000, Gold Medal.
- **Equipment/Brand/Model**: Player equipment, e.g., Rackets and shuttlecocks like Yonex Astrox 99, Li-Ning N90.
- **Federation**: Governing bodies, e.g., BWF (Badminton World Federation), National Badminton Federations.
- **Time/Record**: Performance statistics, e.g., World Record, Personal Best.


### Example Taxonomy Structure for Badminton:

```plaintext
Racket Sports → Sport Category
Sports that involve hitting a shuttlecock or ball with a racket, including badminton, tennis, squash, and others.
└──Badminton Sports
   └── BWF World Championships 2025
       ├── OrganizationAuthority: The governing body organizing the event (e.g., BWF)
       ├── SeasonEdition: Year and edition of the tournament
       ├── TournamentFormat: Knockout, Round Robin, or other format
       ├── CourtSurface: Type of court surface (e.g., Wooden, Synthetic)
       └── DisciplineType
            ├── Singles
                ├── Men's Singles
                    └── DivisionCategoryBracket
                        └── Main Draw
                            ├── MatchRound
                            │   ├── Round of 64
                            │   │   └── Venue: Location or stadium of the round
                            │   ├── Round of 32
                            │   │   └── Venue: Location or stadium of the round
                            │   ├── Quarterfinals
                            │   │   └── Venue: Location or stadium of the round
                            │   ├── Semifinals
                            │   │   └── Venue: Location or stadium of the round
                            │   └── Final
                            │       └── Venue: Location or stadium of the final match
                            ├── PlayerGroup
                            │   └── Players
                            │       ├── PlayerName: Full name of the player
                            │       ├── Country: Country represented by the player
                            │       ├── WorldRanking: BWF World Ranking
                            │       ├── Coach: Name of the coach
                            │       └── TitlesHeld: Relevant titles or championships won
                            ├── Referee: Name or title of match referee
                            ├── Umpires: List of appointed umpires
                            ├── PrizeMoneyRewards: Total prize money distributed in this category
                            └── Broadcast: List of broadcasters or streaming platforms
                └── Women's Singles
            │       └── (Similar structure to Men's Singles)
            └── Doubles
                ├── Men's Doubles
                │   └── (Similar structure adapted for teams of two players)
                ├── Women's Doubles
                │   └── (Similar structure adapted for teams of two players)
                └── Mixed Doubles
                    └── (Similar structure adapted for mixed gender teams)
```
