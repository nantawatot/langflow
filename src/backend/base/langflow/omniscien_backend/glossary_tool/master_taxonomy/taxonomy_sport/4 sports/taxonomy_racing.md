# Racing Taxonomy

```json
{
    "level 1": {
        "category": "Motorsports",
        "description": "Competitive racing involving motor vehicles on circuits or tracks. It includes various disciplines like Formula 1, MotoGP, rallying, and more."
    },
    "level 2": {
        "category": "F1 Racing",
        "description": "The highest class of international single-seater auto racing sanctioned by the FIA, known as Formula 1 or F1 Racing.",
        "reference_website": "https://www.formula1.com/"
    }
}
```
### Core Attributes:
- **Sport Category**: Motorsports, The general category of competitive racing involving motor vehicles.
- **F1 Racing**: The highest class of international single-seater auto racing, known as Formula 1 or F1 Racing.
- **Competition/Event**: Name of the race, e.g., Formula 1 Monaco Grand Prix.
- **Season/Year**: Year the event takes place, e.g., 2023.
- **Round/Stage**: Competition rounds, e.g., Round 1, Round 2, Final Round.
- **Match**: Driver head-to-head competition, e.g., Lewis Hamilton vs Max Verstappen.
- **Driver/Team**: Information about drivers and their teams, e.g., Lewis Hamilton, Max Verstappen.
   - Role/Position: Participant role, e.g., Driver
   - Team: Team names drivers represent, e.g., Mercedes, Red Bull Racing
   - Car Number: Driver’s car number, e.g., 44 (Hamilton), 33 (Verstappen).
   - Nationality: Driver’s country, e.g., United Kingdom, Netherlands.
- **Race Wins**: Driver’s career achievements, e.g., 7-time World Champion (Hamilton).
- **Management**: Team leadership, e.g., Team Principal such as Toto Wolff (Mercedes), Christian Horner (Red Bull Racing).
- **Sponsors**: Sponsors supporting teams and drivers, e.g., Petronas (Mercedes), Infiniti (Red Bull Racing).
- **Venue**: Event location and track details, e.g., Location, Circuit Type (e.g., Street Circuit for Monaco GP).
- **Broadcast**: Media outlets responsible for televising or streaming the event, e.g., Sky Sports, ESPN.



### Example Taxonomy Structure for Racing Sports:
```plaintext
Motorsports → Sport Category
Competitive racing involving motor vehicles on circuits or tracks. It includes various disciplines like Formula 1, MotoGP, rallying, and more.

└── F1 Racing → The highest class of international single-seater auto racing, sanctioned by the FIA. Known for cutting-edge technology and global Grand Prix events.

    └── Competition/Event → Official Formula 1 races or championships.
        ├── Examples → Monaco Grand Prix, British Grand Prix, Italian Grand Prix.
        └── Season/Year → The calendar year or season of the championship (e.g., F1 2025 Season).
        └── Venue → Race location and circuit details.
            ├── Name → Official name of the circuit (e.g., Circuit de Monaco, Silverstone Circuit).
            ├── Circuit Length → Total length of the circuit in kilometers or miles (e.g., 3.337 km for Monaco).
            ├── Number of Laps → Total laps in the Event (e.g., 78 laps for Monaco GP).
            ├── Race Distance → Total distance of the Event (e.g., 260.286 km for Monaco GP).
            ├── Location → Geographical place where the race is held (e.g., Monte Carlo, Silverstone, Monza).
            └── Circuit Type → Style or design of the racing circuit:
                ├── Street Circuit → Temporary circuits in city streets (e.g., Monaco).
                ├── Permanent Circuit → Purpose-built race tracks (e.g., Silverstone).
                └── Hybrid Circuit → Combination of public roads and permanent sections (e.g., Albert Park, Australia).

        └── Team → Racing teams (Full Team Name) competing in the championship .
            ├── Examples → Mercedes-AMG Petronas F1 Team, Oracle Red Bull Racing, Scuderia Ferrari HP, McLaren Formula 1 Team.

            └── Driver → Professional racers under contract with each team.
                ├── Driver Name → e.g., Lewis Hamilton, Max Verstappen, Charles Leclerc.
                ├── Role → Lead Driver, Second Driver, Reserve Driver.
                └── Car Number → Unique identifier for the driver’s car (e.g., 44 for Hamilton, 33 for Verstappen).

            └── Management → Team leadership and strategic decision-makers.
                ├── Team Principal → Overall team lead (e.g., Toto Wolff, Christian Horner).
                └── Technical Director → Oversees car development and performance.

            └── Sponsors → Commercial entities funding the team or drivers.
                ├── Title Sponsors → Main brand associated with the team (e.g., Petronas, Oracle).
                └── Secondary Sponsors → Other commercial partners (e.g., Puma, Mobil 1, TAG Heuer).
            └── Base → Headquarters or base of operations for the team (e.g., Brackley, Milton Keynes, Maranello).
            └── Team Chief → The head of the team, responsible for overall management and strategy.


        └── Regulations → Governing structure and technical rules.
            ├── Governing Body → FIA (Fédération Internationale de l'Automobile).
            ├── Technical Regulations → Rules on car specs, engine limits, weight.
            └── Sporting Regulations → Race format, point system, qualifying rules.

        └── Broadcast → Media rights and coverage.
            ├── TV Networks → Sky Sports F1, ESPN, Canal+, Fox Sports.
            └── Streaming Platforms → F1 TV, DAZN, YouTube highlights.

```