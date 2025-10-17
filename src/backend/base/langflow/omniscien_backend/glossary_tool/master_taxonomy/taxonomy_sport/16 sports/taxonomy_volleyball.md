# Volleyball Taxonomy
```json
{
    "level 1": {
        "category": "Volleyball-Type Sports",
        "description": "sports like volleyball which involve hitting an object back and forth over a net."
    },
    "level 2": {
        "category": "Indoor Volleyball",
        "description": "A team sport played indoors on a hardwood surface, originally developed in 1896 at Springfield College, Massachusetts.",
        "reference_website": "https://www.topendsports.com/sport/list/volleyball.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Volleyball-Type Sports, The general category of sports that includes all types of volleyball competitions and formats.
- **Sport**: Indoor Volleyball, A team sport played indoors on a hardwood surface, originally developed in 1896 at Springfield College, Massachusetts.
  - Indoor Volleyball → Focused Subcategory, The specific variant of volleyball played indoors, typically with six players per side on a standard court.
  - Beach Volleyball → Focused Subcategory, A variant of volleyball played on sand, usually with two players per side.
- **Discipline Type**: Competition formats, e.g., National Team, Club, University, Youth.
- **Competition/Event**: Tournament names, e.g., FIVB Volleyball World Cup, Olympic Games, Volleyball Nations League (VNL).
- **Season/Year**: Year the event takes place, e.g., 2023.
- **Round/Match**: Competition stages, e.g., Preliminary Round, Quarterfinals, Semifinals, Final.
- **Team/Club**: Participating teams, e.g., Brazil, Italy, USA, Russia.
   - **Athlete**: Athlete names, e.g., Karch Kiraly, Giba, Tijana Bošković, Zhu Ting.
     - Role/Position: Athlete roles, e.g., Outside Hitter, Setter, Libero, Opposite.
     - Equipment/Brand/Model: Equipment used, e.g., Volleyball such as Mikasa MVA200, Adidas Volleyball Shoes.
   - **Management**: Team officials, e.g., Team Manager, Head Coach, Assistant.
   - **Sponsors**: Team sponsors, e.g., Nike, Mizuno, Asics.
   - **Ranking**: Team and athlete rankings, e.g., FIVB World Rankings.
- **Venue**: Event locations, e.g., Maracanãzinho (Brazil), Palazzetto dello Sport (Italy).
- **Prize Money/Rewards**: Prize amounts or awards, e.g., $1 million, Gold Medal.
- **Referee/Umpire**: Match officials, e.g., FIVB Umpire, International Referee.
- **Broadcasting/Media**: Media outlets broadcasting matches, e.g., ESPN, FIVB TV, Volleyball TV.
- **Federation**: Organizing bodies, e.g., Fédération Internationale de Volleyball (FIVB), National Federations.


### Example Taxonomy Structure for Volleyball:
```plaintext
Volleyball-Type Sports → Sport Category
A sport category involving teams separated by a net, aiming to ground the ball in the opponent's court.

└── Indoor Volleyball → Focused Subcategory
Traditional volleyball played on an indoor hard court with six players per side.

    └── Discipline Type → Competition formats based on team types.
        ├── National Team
        ├── Club
        ├── University
        └── Youth

        └── Competition/Event → Major official tournaments.
            ├── FIVB Volleyball World Cup
            ├── Olympic Games
            ├── Volleyball Nations League (VNL)
            ├── FIVB Club World Championship
            └── Continental Championships (e.g., CEV, AVC, NORCECA)

            └── Season/Year →
                ├── 2022
                ├── 2023
                ├── 2024
                └── 2025

            └── Venue → Locations where matches are held.
                ├── Maracanãzinho (Brazil)
                ├── Palazzetto dello Sport (Italy)
                ├── Ariake Arena (Japan)
                ├── Atlas Arena (Poland)
                └── Manila Arena (Philippines)

            └── Round/Match → Tournament stages.
                ├── Preliminary Round
                ├── Pool Stage
                ├── Quarterfinals
                ├── Semifinals
                └── Final

            └── Team/Club → Competing teams or professional clubs.
                ├── Brazil
                ├── Italy
                ├── USA
                ├── Russia
                ├── Serbia
                ├── China
                └── Club Teams: Sada Cruzeiro, Zenit Kazan, Imoco Volley

                └── Athlete → Famous or active volleyball players.
                    ├── Karch Kiraly
                    ├── Giba
                    ├── Earvin N'Gapeth
                    ├── Tijana Bošković
                    ├── Zhu Ting
                    ├── Yuji Nishida
                    └── Simone Giannelli

                    └── Role/Position →
                        ├── Outside Hitter
                        ├── Middle Blocker
                        ├── Setter
                        ├── Opposite
                        ├── Libero
                        └── Defensive Specialist

                    └── Equipment/Brand/Model →
                        ├── Ball: Mikasa MVA200, Mikasa V200W
                        ├── Shoes: Mizuno Wave Lightning Z6, Asics Sky Elite FF
                        ├── Jerseys: Nike, Adidas
                        └── Knee Pads, Arm Sleeves, Training Gear

                └── Management →
                    ├── Head Coach
                    ├── Assistant Coach
                    ├── Team Manager
                    ├── Statistician
                    └── Athletic Trainer

                └── Sponsors →
                    ├── Nike
                    ├── Mizuno
                    ├── Asics
                    ├── Mikasa
                    └── Adidas

                └── Ranking →
                    ├── FIVB World Rankings (Men/Women)
                    ├── Olympic Qualification Rankings
                    └── Youth and Junior Rankings

            └── Prize Money/Rewards →
                ├── $1 Million VNL Champions
                ├── Olympic Gold Medal
                ├── MVP of the Tournament
                └── Best Position Awards (e.g., Best Libero, Best Setter)

            └── Referee/Umpire →
                ├── FIVB Umpire
                ├── First Referee
                ├── Second Referee
                ├── Line Judge
                └── Challenge System Operator

            └── Broadcasting/Media →
                ├── ESPN
                ├── Volleyball TV
                ├── FIVB YouTube Channel
                ├── Eurosport
                └── NBC Sports

            └── Federation →
                ├── FIVB (Fédération Internationale de Volleyball)
                ├── CEV (Europe)
                ├── AVC (Asia)
                ├── NORCECA (North/Central America)
                ├── CSV (South America)
                └── National Federations (e.g., USA Volleyball, CBV Brazil)
```