# Handball Taxonomy
```json
{
    "level 1": {
        "category": "Handball Sports",
        "description": "a sport usually played indoors between teams of seven players, who pass a ball to throw it into the goal of the other team. Also known as Team Handball, Olympic Handball, European (Team) handball or Borden Ball. Variations include Beach Handball, Czech Handball and Field Handball."
    },
    "level 2": {
        "category": "Team Handball",
        "description": "which can also be called borden ball, is a type of contact sport in which two teams with seven players each (consisting of six outfield players and one goalkeeper), pass a ball and throw it into the goal of the opposing team.",
        "reference_website": "https://www.topendsports.com/sport/list/handball.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Handball Sports, The general category of the sport that includes all types of handball competitions and formats.
- **Sport**: Team Handball, A fast-paced team sport combining elements of basketball and soccer, played indoors where two teams compete to score goals by throwing a ball into the opponent’s net
- **Focused Subcategory**: Indoor Handball, The specific variant of handball played indoors, typically with seven players per side on a standard court.
- **Focused Subcategory**: Beach Handball, A variant of handball played on sand, usually with five players per side.
- **Discipline Type**: Competition formats, e.g., National League, International Championship, Club Competition.
- **Competition/Event**: Tournament names, e.g., IHF World Handball Championship, EHF Champions League.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Round/Match**: Competition stages, e.g., Quarterfinal, Semifinal, Final.
- **Team/Club**: Participating teams, e.g., Barcelona Handball, THW Kiel.
  - **Athlete**: Player names, e.g., Mikkel Hansen, Nikola Karabatić, Cristina Neagu.
    - **Role/Position**: Player positions, e.g., Goalkeeper, Wing, Pivot, Center Back.
    - **Equipment/Brand/Model**: Player gear, e.g., Ball, Shoes, Apparel such as Adidas Handball Shoes, Molten Ball.
  - **Management**: Team officials, e.g., Coach, Assistant Coach, Team Manager.
  - **Sponsors**: Team sponsors, e.g., Adidas, Puma, Hummel.
- **Referee**: Match officials, e.g., Michael Geiger, Olivier Heintz.
- **Venue**: Event locations, e.g., Lanxess Arena, Palau Blaugrana, Mercedes-Benz Arena.
- **Broadcasting**: Media outlets broadcasting games, e.g., Eurosport, Sky Sports, Handball TV.
- **Federation/Association**: Organizing bodies, e.g., IHF (International Handball Federation), EHF (European Handball Federation).
- **Tournament Format**: Competition styles, e.g., Round-robin, Knockout Stage, Double-elimination.

### Example Taxonomy Structure for Handball:
```plaintext
Handball Sports → Sport Category
A dynamic and fast-paced team sport where players pass and throw a ball with their hands to score goals. Played primarily indoors with 7 players per team, it's popular in Europe and expanding globally.

└── Team Handball → The focused discipline in this taxonomy.
    └── Discipline Type → Major handball competition formats.
        ├── National League → Domestic leagues, e.g., Bundesliga, Liga ASOBAL.
        ├── International Championship → Competitions between national teams, e.g., IHF World Championship, Olympic Games.
        └── Club Competition → Cross-border club events, e.g., EHF Champions League, SEHA League.

        └── Competition/Event → Official tournaments and championships.
            ├── Name → e.g., IHF World Handball Championship, EHF Champions League.
            ├── Season/Year → e.g., 2023, 2024, 2025.

            └── Venue → Match locations.
                ├── Arena/Stadium → e.g., Lanxess Arena (Germany), Palau Blaugrana (Spain), Mercedes-Benz Arena (Germany).
                └── Country/City → e.g., Germany, Denmark, France.

            └── Round/Match → Phases within competitions.
                ├── Group Stage
                ├── Quarterfinal
                ├── Semifinal
                └── Final

            └── Team/Club → Participating national or club teams.
                ├── National Teams → e.g., Denmark, France, Egypt.
                └── Clubs → e.g., FC Barcelona Handbol, THW Kiel, Paris Saint-Germain Handball.

                └── Athlete → Players representing the teams.
                    ├── Name → e.g., Mikkel Hansen, Nikola Karabatić, Cristina Neagu.
                    ├── Role/Position →
                        • Goalkeeper – Defends the goal.
                        • Wing – Attacking player from flanks.
                        • Pivot – Plays near the opponent’s goal area.
                        • Center Back – Organizes attack and passes.

                    └── Equipment/Brand/Model →
                        ├── Ball → e.g., Molten, Select.
                        ├── Shoes → e.g., Adidas Handball Spezial, Kempa Wing.
                        └── Apparel → Jerseys, shorts, protective gear.

                ├── Management →
                    ├── Head Coach
                    ├── Assistant Coach
                    ├── Team Manager
                    └── Physiotherapist

                ├── Sponsors →
                    ├── Apparel Sponsors → e.g., Adidas, Puma, Hummel.
                    └── Corporate Sponsors → e.g., Lidl, Velux.

            └── Referee →
                ├── Officials → e.g., Michael Geiger, Olivier Heintz.
                └── Referee Roles → On-field Referee, Technical Delegate.

            └── Broadcasting →
                ├── TV → e.g., Eurosport, Sky Sports.
                └── Online → e.g., Handball TV, EHF TV.

            └── Federation/Association →
                ├── IHF → International Handball Federation.
                └── EHF → European Handball Federation.

            └── Tournament Format →
                ├── Round-robin
                ├── Knockout Stage
                └── Double-elimination
