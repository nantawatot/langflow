# Cricket Sports Taxonomy
```json
{
    "level 1": {
        "category": "Cricket Sports",
        "description": "a team sport played on a rectangular pitch in the center of a large grass oval, two batters protect their wicket while the fielding team attempt to get them out. Forms include Test, One-Day and T20."
    },
    "level 2": {
        "category": "Test Cricket",
        "description": "the longest form of cricket, played over 5 days.",
        "reference_website": "https://www.topendsports.com/sport/list/cricket-test.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Cricket Sports, The general category of the sport that includes all types of cricket competitions and formats.
- **Sport**: Cricket, A bat-and-ball team sport played internationally, featuring formats that range from multi-day matches to fast-paced limited overs games. It is popular globally, especially in countries like India, England, Australia, and others.
- **Test Cricket**: The traditional and longest form of cricket, played over multiple days, focusing on endurance and strategy.
- **Discipline Type**: Competition formats, e.g., Test Match, One Day International (ODI), Twenty20 (T20).
- **Competition/Event**: Tournament names, e.g., ICC World Cup, Ashes Series, IPL (Indian Premier League).
- **Season/Year**: Year the event occurs, e.g., 2023, 2024.
- **Stage/Match/Round**: Competition phases, e.g., Semi-final, Final, Group Stage.
- **Team/Club**: Participating teams, e.g., India, Australia, England, Mumbai Indians.
  - **Athlete**: Player names, e.g., Virat Kohli, Steve Smith, Ben Stokes, Sachin Tendulkar.
    - **Role/Position**: Player roles, e.g., Batsman, Bowler, All-rounder, Wicketkeeper.
    - **Equipment/Brand/Model**: Player gear, e.g., Gray-Nicolls Bat, Kookaburra Ball, Adidas Cricket Shoes.
  - **Management**: Team officials, e.g., Coach, Captain, Team Manager, Batting Coach.
  - **Sponsors**: Team sponsors, e.g., Nike, Puma, Paytm, Vivo, Audi.
- **Title/Championship**: Championship titles, e.g., ICC Champions Trophy, IPL Title.
- **Venue/Location**: Event locations, e.g., Lord’s Cricket Ground, MCG (Melbourne Cricket Ground), Eden Gardens.
- **Match Result**: Outcomes, e.g., Win, Loss, Draw, No Result, Tie.
- **Cricket Rules**: Key rules, e.g., LBW (Leg Before Wicket), Boundary, Over, Duckworth-Lewis Method.
- **Umpire/Referee**: Officials, e.g., On-field Umpire, Third Umpire, Match Referee.

### Example Taxonomy Structure for Cricket:
```plaintext
Cricket Sports → Sport Category
A global bat-and-ball sport played between two teams. It includes long-format games like Test cricket and short formats like ODIs and T20s.

└── Format Type → Main formats in professional cricket.
    ├── Test Cricket → Traditional 5-day matches emphasizing technique and endurance.
    ├── One Day International (ODI) → 50-over limited-overs format.
    └── Twenty20 (T20) → 20-over fast-paced matches designed for entertainment.

    └── Discipline Type → Competitive classifications based on match format.
        ├── Test Match → Played over five days with two innings per team.
        ├── ODI Match → Played over a single day, 50 overs per side.
        └── T20 Match → Short, dynamic games, 20 overs per side.

        └── Competition/Event → Official series and tournaments.
            ├── Name → e.g., ICC World Cup, Ashes Series, Indian Premier League (IPL), T20 World Cup.
            ├── Season/Year → e.g., 2023, 2024, 2025.

            └── Venue/Location → Places where matches are hosted.
                ├── Stadium → e.g., Lord’s, Melbourne Cricket Ground, Eden Gardens, Wankhede Stadium.
                └── Country/City → e.g., India, England, Australia, South Africa.

            └── Team/Club → National, domestic, or franchise teams.
                ├── National Teams → e.g., India, England, Australia, Pakistan.
                └── Franchise Clubs → e.g., Mumbai Indians, Chennai Super Kings, Sydney Thunder.

                └── Athlete → Players participating in competitions.
                    ├── Name → e.g., Virat Kohli, Steve Smith, Ben Stokes, Rashid Khan.
                    ├── Role/Position →
                        • Batsman – Specialist in scoring runs.
                        • Bowler – Specialist in dismissing batsmen.
                        • All-rounder – Contributes in both batting and bowling.
                        • Wicketkeeper – Fielder positioned behind the stumps.

                    └── Equipment/Brand/Model → Player gear.
                        ├── Bat → e.g., Gray-Nicolls, SG, SS Ton.
                        ├── Ball → e.g., Kookaburra, Dukes, SG Test.
                        ├── Protective Gear → Helmets, Pads, Gloves.
                        └── Footwear → e.g., Adidas, Puma, New Balance.

                ├── Management → Team leadership and support staff.
                    ├── Coach
                    ├── Captain
                    ├── Batting Coach
                    └── Team Manager

                ├── Sponsors → Corporate entities supporting teams or players.
                    ├── Apparel/Kit Sponsors → e.g., Nike, Adidas, Puma.
                    ├── Financial/Telecom → e.g., Paytm, Vivo, Airtel.
                    └── Other Brands → e.g., BYJU’S, Dream11, Audi.

            └── Title/Championship → Prizes awarded in events.
                ├── ICC World Cup Trophy
                ├── T20 World Cup
                ├── ICC Champions Trophy
                └── IPL Title

            └── Umpire/Referee → Match officials overseeing rules and conduct.
                ├── On-field Umpire → Positioned on the field.
                ├── Third Umpire → Uses video technology for decisions.
                └── Match Referee → Ensures fair play and code of conduct.

```