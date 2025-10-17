# Rugby Sport Taxonomy
```json
{
    "level 1": {
        "category": "Football Sports",
        "description": "there are many sports that are called 'football' such as American football, Association football (soccer), Rugby and Aussie Rules, but equally many variations of these such as beach and indoor soccer, touch rugby and flag football."
    },
    "level 2": {
        "category": "Rugby Union",
        "description": "A form of rugby football similar to Rugby League but with different rules, such as contested scrums and lineouts.",
        "reference_website": "https://www.topendsports.com/sport/list/rugby-union.htm"
    }
}
```


### Core Attributes:
- **Sport Category**: Football Sports, The general category of the sport that includes all types of rugby competitions and formats.
- **Sport**: Rugby, A contact team sport that originated in England, played with an oval ball, where two teams compete to score points by carrying, passing, and kicking the ball.
- **Rugby Union**: A specific code of rugby with 15 players per team, known for its complex rules and set pieces like scrums and lineouts.
- **Rugby League**: A variant of rugby with 13 players per team, emphasizing speed and fewer stoppages compared to Rugby Union.
- **Discipline Type**: Competition formats, e.g., International, Club, Sevens, Super Rugby.
- **Competition/Event**: Tournament names, e.g., Rugby World Cup, Six Nations Championship, Super Rugby.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Match/Stage**: Competition stages, e.g., Quarter-finals, Semi-finals, Final.
- **Team/Club**: Participating teams, e.g., New Zealand All Blacks, South Africa Springboks, England Rugby.
  - **Athlete**: Player names, e.g., Richie McCaw, Jonny Wilkinson, Siya Kolissi.
    - Role/Position: Player positions, e.g. , Fly-half, Lock, Scrum-half, Fullback, Prop.
  - **Management**: Team officials, e.g., Head Coach, Assistant Coach, Team Manager.
  - **Sponsors**: Team Sponsor, e.g., Aidas, Coca-Cola, Emirates.
  - **Ranking**: Team rankings, e.g., World Rugby Rankings, Six Nations Standings
  - **Fanbase**: Team fan groups, e.g., "All Blacks Nation," "Springboks Supporters"
- **Referee**: Match officials, e.g., Nigel Owens, Jerome Garces.
- **Venue**: Event locations, e.g., Twickenham, Eden Park, Stade de France
- **League**: Related leagues, e.g., Rugby Preniership, Top 14, Supe Rugby
- **Broadcasting**: Media outlets broadcasting events, e.g., ESPN, Fox Sports, DAZN.


### Example Taxonomy Structure for Rugby:

```plaintext
Football Sports → Sport Category
A broad category encompassing various forms of football, including contact and non-contact formats played globally.

└── Rugby Union → A rugby code played with 15 players per side, known for its international competitions and club leagues.

    └── Discipline Type → Competition formats within rugby union.
        ├── International Tests → Matches between national teams.
        ├── Club Leagues → Domestic and regional club competitions.
        ├── Sevens → Fast-paced 7-a-side format.
        └── Super Rugby → Elite Southern Hemisphere professional club tournament.

        └── Competition/Event → Official tournaments and leagues.
            ├── Rugby World Cup
            ├── Six Nations Championship
            ├── The Rugby Championship
            ├── Super Rugby
            └── HSBC World Rugby Sevens Series

            └── Venue →
                ├── Twickenham Stadium (England)
                ├── Eden Park (New Zealand)
                ├── Stade de France (France)
                └── Suncorp Stadium (Australia)

            └── Season/Year → e.g., 2023, 2024

            └── Match/Stage →
                ├── Pool Stage
                ├── Quarter-finals
                ├── Semi-finals
                └── Final

            └── Team/Club → National or club teams.
                ├── New Zealand All Blacks
                ├── South Africa Springboks
                ├── England Rugby
                ├── Ireland Rugby
                ├── Crusaders (NZ)
                └── Leinster Rugby (Ireland)

                └── Athlete → Player participants.
                    ├── Richie McCaw
                    ├── Jonny Wilkinson
                    ├── Siya Kolisi
                    └── Antoine Dupont

                    └── Role/Position →
                        ├── Fly-half
                        ├── Scrum-half
                        ├── Fullback
                        ├── Lock
                        ├── Prop
                        ├── Hooker
                        ├── Flanker
                        └── Number Eight

                    └── Equipment/Brand/Model →
                        ├── Jerseys (Nike, Canterbury)
                        ├── Cleats (Adidas, Puma)
                        ├── Headgear
                        └── Sunglasses (Oakley)

                └── Management →
                    ├── Head Coach
                    ├── Assistant Coach
                    ├── Team Manager
                    └── Medical Staff

                └── Sponsors →
                    ├── Adidas
                    ├── Coca-Cola
                    ├── Emirates
                    └── DHL

                └── Ranking →
                    ├── World Rugby Rankings
                    ├── Six Nations Standings
                    └── The Rugby Championship Table

                └── Fanbase →
                    ├── All Blacks Nation
                    ├── Springboks Supporters
                    ├── England Rugby Fans
                    └── Irish Rugby Faithful

            └── Referee →
                ├── Nigel Owens
                ├── Jerome Garces
                ├── Wayne Barnes
                └── Jaco Peyper

            └── Award →
                ├── World Rugby Player of the Year
                ├── Best Try of the Tournament
                ├── Coach of the Year
                └── Breakthrough Player Award

```