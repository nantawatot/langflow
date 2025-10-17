# Horse Racing Taxonomy
```json
{
    "level 1": {
        "category": "Horse Sports",
        "description": "such as polo, equestrian and horse racing."
    },
    "level 2": {
        "category": "Horse Racing",
        "description": "equestrian sport that involves jockeys riding horses or being pulled along by horses.",
        "reference_website": "https://www.topendsports.com/sport/list/horse-racing.htm"
    }
}
```

### Core Attributes:
- **Sport Category**: Horse Sports, The general category of the sport that includes all types of horse racing competitions and formats.
- **Sport**: Horse Racing, A competitive equestrian sport where" jockeys ride horses or are pulled along by horses. It includes various formats and events, such as flat racing and jump racing.
- **Discipline Type**: Horse Racing, The specific equestrian discipline focused on here.
  - **Flat Racing**: A race where horses run on a level track without obstacles. It is the most common form of horse racing.
  - **Jump Racing**: Races that involve jumping over obstacles such as fences or hurdles.
  - **Harness Racing**: A form of horse racing where horses pull a two-wheeled cart called a sulky, and the driver sits in the cart.
  - **Endurance Racing**: Long-distance.
- **Competition/Event**:  Race names, e.g., The Kentucky Derby, The Grand National.
- **Season/Year**: Year the event occurs, e.g., 2023.
- **Race/Heat**: Race rounds within an event daty, e.g., Race 1, Race 2, Final Race.
- **Horse/Runner**: Names of horses or competitors, e.g., Secretariat, Black Caviar, Red Rum.
  - **Jockey**: Names of riders, e.g., Frankie Dettori, John Velazquez, Ruby Walsh.
  - **Trainer**: Names of trainers, e.g., Bob Baffert, Aidan O'Brien, Gordon Elliott.
  - **Owner**: Horse owners, e.g., Sheikh Mohammed bin Rashid Al Maktoum, Coolmore Stud.
  - **Team/Stable**: Names of stables or training teams, e.g., Godolphin, Ballydoyle.
  - **Role/Position**: Roles of participants, e.g., Jockey, Trainer, Owner.
- **Track/Venue**: Race locations, e.g., Churchill Downs, Ascot, Aintree.
- **Betting Odds**: Betting odds, e.g., 5/1, 10/1.
- **Organization**: Event officials and coordinators like Race Director, Event Manager, Steward
- **Sponsors**: Sponsors supporting events, e.g., Lexus, Rolex.
- **Prize Money/Rewards**: Financial and trophy rewards, e.g., $2 million, £1 million
- **Broadcast**: Media outlets responsible for televising or streaming the event, e.g., NBC Sports, Sky Sports.


### Example Taxonomy Structure for Horse Racing:
```plaintext
Horse Sports → Sport Category
Equestrian sports where horse and rider collaborate in speed, agility, and control. Horse racing is a globally popular discipline with historic races and significant betting interest.

└── Horse Racing → The focused equestrian discipline in this taxonomy.
    └── Discipline Type → Main formats of horse racing.
        ├── Flat Racing → Races on a level track without obstacles.
        └── Jump Racing → Races involving hurdles or steeplechase fences.

        └── Competition/Event → Prestigious race events around the world.
            ├── Name → e.g., The Kentucky Derby, The Grand National, Melbourne Cup.
            ├── Season/Year → e.g., 2023, 2024.

            └── Track/Venue → Racecourses where events are held.
                ├── Name → e.g., Churchill Downs, Ascot, Aintree.
                └── Location → Country/City of the venue.

            └── Race/Heat → Specific runs within the event.
                ├── Race 1, Race 2, Heat A, Final Race, etc.
                └── Distance → e.g., 1.25 miles, 3200 meters.

            └── Horse/Runner → Equine athletes participating in races.
                ├── Name → e.g., Secretariat, Red Rum, Winx, Black Caviar.

                └── Jockey → Professional riders.
                    ├── Name → e.g., Frankie Dettori, John Velazquez, Ruby Walsh.
                    └── Nationality, Career Wins, Experience.

                └── Trainer → Horse preparation experts.
                    ├── Name → e.g., Bob Baffert, Aidan O'Brien, Gordon Elliott.
                    └── Training Facility/Location.

                └── Owner → Individuals or groups who own the horse.
                    ├── Name → e.g., Sheikh Mohammed bin Rashid Al Maktoum, Coolmore Stud.
                    └── Ownership Type → Private, Syndicate.

                └── Team/Stable → Organized entities managing horses.
                    ├── Name → e.g., Godolphin, Ballydoyle, Juddmonte Farms.
                    └── Country/Region.

                └── Betting Odds →
                    ├── Odds Format → e.g., 5/1, 10/1, 2.50 (decimal).
                    └── Source → Official betting agencies.

            └── Management →
                ├── Race Director
                ├── Event Coordinator
                ├── Steward
                └── Veterinarian

            └── Sponsors →
                ├── Corporate Sponsors → e.g., Lexus, Rolex, Emirates.
                └── Local Partners → e.g., banks, media outlets.

            └── Prize Money/Rewards →
                ├── Total Purse → e.g., $2 million, £1 million.
                ├── Distribution → Winner, Place, Show payouts.
                └── Trophy/Awards → Cups, plaques, garlands.
```