# Tennis Taxonomy
```json
{
    "level 1": {
        "category": "Racket Sports",
        "description": "such as tennis, squash, badminton and table tennis."
    },
    "level 2": {
        "category": "Tennis",
        "description": "is one of the most popular individual sports in the world.",
        "reference_website": "https://www.topendsports.com/sport/list/tennis.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Racket Sports, The general category of sports that involve hitting a ball with a racket.
- **Sport**: Tennis, A widely recognized individual sport played on various surfaces.
- **Discipline Type**: Singles, Doubles, Mixed Doubles, The different formats of tennis matches.
- **Competition/Event**: Tournament names, e.g., Wimbledon, Australian Open, US Open.
- **Season/Year**: Year the event takes place, e.g., 2023.
- **Round/Match**: Competition stages, e.g., First Round, Quarterfinals, Semifinals, Final.
- **Player/Athlete**: Player names, e.g., Novak Djokovic, Rafael Nadal, Serena Williams, Naomi Osaka.
   - **Role/Position**: Player roles or statuses, e.g., World No. 1, Top Seed, Wild Card.
   - **Equipment/Brand/Model**: Player equipment, e.g., Rackets like Wilson Blade 98, Babolat Pure Drive.
   - **Management**: Team officials, e.g., Coach, Agent, Tournament Director.
   - **Sponsors**: Player sponsors, e.g., Nike, Rolex, Wilson.
   - **Ranking**: Player rankings, e.g., ATP Rankings, WTA Rankings.
   - **Team/Club**: Training academies or clubs, e.g., Rafael Nadal Academy, Mouratoglou Tennis Academy.

- **Venue**: Tournament locations, e.g., Melbourne Park, Roland Garros, All England Club.
- **Prize Money/Rewards**: Awards and prize amounts, e.g., $2.5 million, £1 million.
- **Federation**: Governing bodies, e.g., ITF (International Tennis Federation), ATP (Association of Tennis Professionals
- **Broadcasting/Media**: Media outlets broadcasting events, e.g., ESPN, Eurosport, Tennis Channel.


### Example Taxonomy Structure for Tennis:
```plaintext
Racket Sports → Sport Category
A category of sports played with rackets and a ball or shuttlecock, focusing on individual or team competition.

└── Tennis → Focused sub-sport category
A global racket sport played on various surfaces in singles, doubles, and mixed formats.

    └── Discipline Type → Formats of play.
        ├── Singles
        ├── Doubles
        └── Mixed Doubles

        └── Competition/Event → Official tournaments.
            ├── Wimbledon
            ├── Australian Open
            ├── US Open
            ├── Roland Garros (French Open)
            ├── ATP Finals
            ├── WTA Finals
            └── Davis Cup / Billie Jean King Cup

            └── Season/Year → e.g., 2023, 2024

            └── Venue →
                ├── Melbourne Park (Australia)
                ├── Roland Garros (France)
                ├── All England Club (UK)
                ├── USTA Billie Jean King National Tennis Center (USA)
                └── O2 Arena (UK)

            └── Court Surface →
                ├── Grass
                ├── Clay
                └── Hard Court

            └── Round/Match →
                ├── First Round
                ├── Second Round
                ├── Quarterfinals
                ├── Semifinals
                └── Final

            └── Team/Club → Training entities or academies.
                ├── Rafael Nadal Academy
                ├── Mouratoglou Tennis Academy
                ├── IMG Academy
                └── Saddlebrook Tennis Academy

                └── Player/Athlete → Tennis professionals.
                    ├── Novak Djokovic
                    ├── Rafael Nadal
                    ├── Serena Williams
                    ├── Naomi Osaka
                    ├── Carlos Alcaraz
                    └── Iga Świątek

                    └── Role/Position →
                        ├── World No. 1
                        ├── Top Seed
                        ├── Wild Card
                        ├── Qualifier
                        └── Lucky Loser

                    └── Equipment/Brand/Model →
                        ├── Rackets: Wilson Blade 98, Babolat Pure Drive, Head Speed Pro
                        ├── Shoes: Nike Air Zoom Vapor, Adidas Barricade
                        └── Apparel: Uniqlo, Nike, Asics

                    └── Ranking →
                        ├── ATP Rankings
                        └── WTA Rankings

                └── Management →
                    ├── Coach
                    ├── Agent
                    ├── Physiotherapist
                    └── Tournament Director

                └── Sponsors →
                    ├── Nike
                    ├── Adidas
                    ├── Rolex
                    ├── Wilson
                    ├── Head
                    └── Lacoste

            └── Prize Money/Rewards →
                ├── Wimbledon Winner: £2.5 million
                ├── US Open Winner: $3 million
                ├── Australian Open Winner: AUD $2.9 million
                └── Roland Garros Winner: €2.3 million

            └── Broadcasting/Media →
                ├── ESPN
                ├── Eurosport
                ├── Tennis Channel
                ├── Amazon Prime Video
                └── Sky Sports

            └── Federation →
                ├── ITF (International Tennis Federation)
                ├── ATP (Association of Tennis Professionals)
                └── WTA (Women's Tennis Association)















