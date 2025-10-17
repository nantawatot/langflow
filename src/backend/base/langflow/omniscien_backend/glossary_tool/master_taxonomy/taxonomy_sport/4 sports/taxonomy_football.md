# Football Taxonomy

### Core Attributes:
- **Sport Category**: Football, The general category of the sport that includes all types of football games globally.
- **Sport**: Association Football, The most widely recognized form of football (commonly known as “soccer” in some countries). It is governed by FIFA and played with two teams of 11 players each using a round ball on a rectangular field.
- **Discipline Type**: The format or structure of the football competition. Common types include:
    - League Tournament – A season-based competition where teams accumulate points over multiple matches (e.g., Premier League).
    - Knockout Tournament – A format where teams compete in elimination rounds, with losing teams being removed from the competition (e.g., FA Cup).
    - Penalty Shootout – A tie-breaking method where teams take alternate penalty kicks to determine a winner in case of a draw after regular and extra time.
- **Competition/Event**: The official name of a football tournament or event organized by national or international governing bodies. Examples include the FIFA World Cup, UEFA Euro, Copa America, and Champions League.
- **Season/Year**: The year or season in which the competition is held. This is useful for organizing data chronologically, e.g., FIFA World Cup 2022.
- **Round/Stage**: The phase of the tournament within the event structure. Common examples include:
    - Group Stage – Teams are split into groups and play round-robin matches.
    - Knockout Stage – Teams compete in elimination rounds, leading to the final.
    - Final – The last match that determines the champion of the tournament.
- **Match**: A single football game between two teams. Typically referenced by the names of the competing teams (e.g., Brazil vs Serbia), along with its result, date, and venue.
- **Team/Club**: The name of a national or club team participating in the match. For example, national teams like Brazil, Serbia, or club teams like Manchester United, Barcelona.
- **Athlete**: A football player who participates in matches as part of a team. Examples include Neymar, Dušan Tadić, and Lionel Messi.
- **Role/Position**: The position a player holds on the field, which defines their responsibilities. Examples include:
    - Forward – An attacking role focused on scoring goals.
    - Midfielder – Controls the flow between attack and defense.
    - Goalkeeper – Prevents the opposing team from scoring.
    - Defender – Protects their goal by stopping opposing attackers.
- **Team Captain**: A designation that indicates whether a player is the captain of the team. The captain often leads the team on the field, communicates with referees, and represents the team in official events. (e.g., true if the player is captain)
- **Club**: The professional football club a player is contracted with. This is separate from national team duty. Examples include Paris Saint-Germain, Ajax, and Manchester City.
- **National Team**: The national team that the player represents in international competitions. Examples include Brazil, Serbia, France, and Argentina.
- **Management**: The coaching and leadership staff of a football team. Key roles include:
    - Head Coach – Responsible for tactical decisions, team selection, and match-day strategy. Examples include Tite (Brazil’s coach) and Dragan Stojković (Serbia’s coach).
    - Assistant Coach – Supports the head coach in training and strategy.
- **Sponsors**: Companies or organizations that financially support a team, event, or player. Sponsors may appear on team kits, advertising, and promotional material. Examples include Nike, Budweiser, Adidas, and Gazprom.
- **Broadcast**: The media outlets responsible for televising or streaming the event, such as ESPN, Sky Sports, or beIN Sports.
- **Venue**: The location where the match is played, which can be a stadium or arena. Examples include Wembley Stadium, Camp Nou, and Old Trafford.
- **Prize Money/Rewards**: Awards and prize amounts given to teams or players for their performance in competitions. This can include trophies, medals, and financial rewards.


### Example Taxonomy Structure for Football:
```plaintext
Football → Sport Category
A global team sport where two teams compete to score goals by moving a ball into the opponent’s net. It includes various formats, with Association Football being the most popular.

└── Association Football → The most widely recognized form of football (soccer), governed by FIFA. Played with 11 players per side. Excludes formats like American football and futsal.

    └── Discipline Type → Defines the structure or format of the competition:
        ├── League Tournament → Season-based competition with point accumulation over several months (e.g., Premier League, La Liga).
        ├── Knockout Tournament → Elimination-based format; losing teams are removed from the competition (e.g., UEFA Champions League, FA Cup).
        └── Penalty Shootout → Tie-breaking method where teams take alternate penalty kicks to decide the winner of a drawn match.

        └── Competition/Event → The official name of a football tournament or event.
            ├── FIFA World Cup → A global tournament held every four years featuring national teams.
            ├── UEFA Euro → European Championship tournament for national teams in Europe.
            └── Copa America → South American national team championship.

            └── Round/Stage → Phases within a competition that organize team progression.
                ├── Group Stage → Teams divided into groups and play round-robin matches.
                ├── Knockout Stage → Single elimination rounds (e.g., Round of 16, Quarterfinals, Semifinals).
                └── Final → The ultimate championship match that determines the winner.

                └── Match → A single football game played between two teams.
                    ├── Match Name → Example: Brazil vs Serbia.
                    ├── Date → Scheduled date of the match (e.g., 24 November 2022).
                    ├── Venue → Stadium or location where the match is held (e.g., Lusail Stadium, Qatar).
                    ├── Score → Final result of the match (e.g., 2-0).
                    └── Referees → Official match referees and assistants (e.g., Referee: Michael Oliver).

            └── Team/Club → An official football team participating in the competition.
                ├── Team Name → Example: Brazil National Team, Manchester City FC.
                ├── Country/League Affiliation → National or club-level identification.
                ├── Squad/Player Roster → List of athletes selected to compete.

                └── Athlete → A professional football player representing a team.
                    ├── Full Name → Example: Neymar Jr.
                    ├── Jersey Number → Player’s shirt number during the event (e.g., #10).
                    ├── Position → Player's field role (e.g., Forward, Midfielder, Goalkeeper).
                    ├── Age/Nationality → Biographical details (e.g., 31, Brazil).
                    └── Stats → Match-specific or tournament-level performance data (e.g., Goals, Assists, Pass Completion Rate).

                └── Management → Leadership and coaching staff of the team.
                    ├── Head Coach → Main strategist and team leader (e.g., Didier Deschamps).
                    └── Assistant Coach → Supports head coach with tactics and training (e.g., Guy Stéphan).

                └── Sponsors → Companies or brands supporting the team.
                    ├── Apparel Sponsor → Provider of team kits (e.g., Nike, Adidas).
                    └── Commercial Sponsors → Additional supporters (e.g., Budweiser, Gazprom).
           └── Broadcast → Media outlets covering the event.
                ├── Television Networks → Channels broadcasting the matches (e.g., ESPN, Sky Sports).
                └── Streaming Services → Online platforms providing live coverage (e.g., DAZN, FuboTV).

            └── Venue → The location where the match is played.
                ├── Stadium Name → Official name of the stadium (e.g., Lusail Stadium).
                ├── City/Country → Geographic location of the venue (e.g., Lusail, Qatar).
                └── Capacity → Seating capacity of the stadium (e.g., 88,966).

            └── Prize Money/Rewards → Financial and symbolic awards for performance.
                ├── Trophy → Physical award for winning teams (e.g., FIFA World Cup Trophy).
                ├── Medals → Awards for top-performing teams (e.g., Gold, Silver, Bronze).
                └── Financial Rewards → Monetary prizes distributed to teams and players.
```