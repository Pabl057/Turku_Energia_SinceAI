Challenge Brief
Template for challenge partners · Since AI Hackathon 2026 · 6–8 Nov 2026, Turku

This template helps you define a clear, buildable challenge for the hackathon. Fill in each section below
— the more concrete the problem and the evaluation criteria, the better the prototypes you receive. You
can return it to us as is, and we are happy to help shape it into a ready event challenge.
Questions about the template? Riku Lauttia · riku.lauttia@sinceai.fi · +358 40 938 8820

Challenge partner (company)
Turku Energia

01 Challenge Title
A short, memorable name for the challenge. Example: “Automated support ticket routing.”

Anticipate power system imbalance: build an imbalance price forecast

02 Problem Statement
What business or technical problem should be solved? Describe the context, why it matters, and the main
pain points.
Keeping the power system in balance requires production and consumption to be matched
continuously. Electricity can hardly be stored in the grid, so an imbalance shows up immediately as
a deviation in grid frequency. The power system corrects normal deviations automatically with
reserves, i.e. production or consumption capacity that power system actors have reserved in
advance for use, so the frequency deviation itself usually goes unnoticed by an ordinary electricity
user. It only becomes a problem if the deviation grows so large that the operation of the power
system has to be safeguarded, for example by disconnecting consumption.
As the share of wind and solar power grows, the balancing need of the power system varies ever
more rapidly, which shows up as significant fluctuations in imbalance prices. An imbalance is the
difference between planned and actual electricity production or consumption, in practice a
forecast error. If a company has planned to consume 100 units but consumes 110, an imbalance of
10 units arises.
For market parties, i.e. companies and other actors that buy, sell, produce or consume electricity
on the electricity market, anticipating the imbalance is challenging because the price is affected
simultaneously by several factors, such as consumption, weather conditions, renewable
production forecasts, the transmission situation in the grid, the availability of reserves and the
momentary balancing need of the market. The available information is scattered across different
sources, and it is not easy to form an overall picture of the upcoming market situation from it.
Market parties trade on the previous day in Day-Ahead trading (the trading venue is a power
exchange, for example Nord Pool) according to their consumption and production forecasts. When
the forecasts are updated during the day, the forecast changes must be traded on the intraday

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 1

markets, i.e. Intraday trading. If the forecast error is not corrected, the balance responsible
party/market party is obliged to procure the missing electricity at the imbalance price. In other
words, if the forecast error is not corrected through intraday trading or with own resources, the
remaining imbalance is settled in imbalance settlement at the imbalance price. Because, for
example, electricity sales are based on the spot price, i.e. the bidding zone price formed on the
day-ahead market, the price difference causes a cost to the market party, which has to be
managed through well-timed and well-priced trading or by adjusting own assets (production or
consumption resources).
Limited visibility into future balancing energy prices increases the risk of imbalance costs and
makes it harder to plan trading, production and demand response. Decisions often have to be
made under uncertainty, without a reliable estimate of how the market will develop over the
coming hours.
In this challenge, the team's goal is to find out whether historical market, production, consumption
and weather data can be used to build an AI-based forecasting model that
-

produces a well-reasoned estimate of the imbalance price for the next 10 hours,

-

identifies the most important factors affecting the price, and

-

describes the uncertainty associated with the forecast.

What makes up the price to be forecast?
Electricity production and consumption are balanced with two balancing mechanisms: mFRR and
aFRR. aFRR is the automatically activated frequency restoration reserve of the power system and
mFRR the manually activated frequency restoration reserve. They respond in different ways and
are handled on different markets. Their energy prices affect the imbalance price, which is
determined separately for each 15-minute period.
The balancing direction is decisive in the price calculation: is more production or less consumption
needed (upward regulation), or the opposite (downward regulation)? The dominant direction is
determined on the basis of mFRR balancing. In practice, balancing steers the price: in upward
regulation a higher price encourages increasing production or reducing consumption, whereas in
downward regulation a lower price encourages reducing production or increasing consumption.
Thus, if more production (or less consumption) is wanted, the price is higher than the Day-Ahead
price. If less production (or more consumption) is wanted, the balancing price is lower than the
Day-Ahead price.
From 1 June 2026, when the mFRR activated in Finland and the fulfilled balancing need of aFRR are
both in the dominant direction, the price is calculated as an average weighted by their balancing
volumes. In practice, the larger balancing volume gets a larger weight in the calculation – so this is
not a sum of the prices.
If a quarter-hour has no dominant mFRR direction, the price is based on the day-ahead price, i.e.
the spot price, which is formed on the electricity market of the previous day. This can happen even
if aFRR balancing has been used. In other situations, separate pricing rules apply.
With the solution, market actors can better anticipate the balancing need of the grid and make
informed operational decisions. In other words, as a scope boundary: the task is to produce
forecasts, observations and decision support for a person working at Turku Energia in production
or electricity trading, for example as a production shift supervisor or an electricity trading
specialist.
Turku Energia's role: Turku Energia plans electricity procurement, trading and the operation of its
available production and consumption resources based on market forecasts. When forecasts
change, the company has to assess whether trading or the production plan needs to be changed

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 2

and whether it is worthwhile to offer available flexible capacity to the reserve markets. Better
visibility into the balancing need of the system and into future prices would help identify significant
situations earlier and support timely decision-making.
The concrete deliverables are:
Level 1: mandatory forecast data
The output produces, for each upcoming 15-minute period over the next 10 hours:
 the forecast imbalance price
 an uncertainty interval for the forecast
 an estimate of the direction of the system balancing need
 a probability (0–1), separately, of an exceptionally high and of an exceptionally low price
(exceptional = the imbalance price deviates from the spot price by more than 50 €/MWh
upwards or downwards)
 the most important factors affecting the forecast
 information on when the forecast was made and how fresh the data used is.
The result is produced as a machine-readable time series in the CSV format described in the
starter package. The solution may additionally provide the result through an API, but no production
integration needs to be built.
Level 2: situational overview for the user
From the forecast data, the solution surfaces only the situations that are relevant for decisionmaking, for example:
 the risk of a price spike has increased
 the system is expected to need upward or downward regulation
 the forecast has changed materially since the previous update
 different data sources or models give a conflicting signal
 the situation is worth an expert review.
An example message could be:
The risk of an exceptionally high price has increased for 16:00–18:00. The forecast
is driven in particular by a reduced wind power forecast, increased consumption
and the cross-border transmission situation. The forecast involves high uncertainty,
so it is worth checking the production plan and the market position.
The solution may also give recommendations for action, but in the first phase as suggestions to
review – not as automatic trading or operating instructions, because open market data alone does
not tell, for example:
 what capacity Turku Energia has available at that moment
 what the marginal cost of the plant is
 how quickly the output of the plant can be changed
 what heat, contractual or ownership constraints apply to the operation
 whether the capacity is already committed to another market.
Therefore, a solution based on open data can reliably say “check this situation”, but not yet with
certainty “offer this amount from this plant to this market”. An actual capacity recommendation is
a possible natural next development step, in which Turku Energia's internal plant and cost data is
brought in.
The result is a user interface/page that presents this information with the necessary graphical
elements, plus an API from which this information can be retrieved for display.

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 3

03 Guiding Question
One question that focuses the challenge. Example: “How might we use AI to predict equipment failures before
they happen?”
How can AI help anticipate the balancing need of the power system and the imbalance price
accurately enough to support operational decision-making?

04 Recommended AI Approaches Optional
Any relevant technologies, models or methods — for example NLP for text analysis, classification, knowledge
graphs, agents, or RAG.
Methods, models and technologies the team considers best

05 Dataset or API Provided If any
Describe any data or resources you provide: format, a link if ready, and sample size. If none, state what
builders should expect.
At Since AI, challenges can be built with synthetic, anonymized or public data — no personal data is required.

The challenge uses publicly available market, production and consumption data as well as the spot
price licensed by Turku Energia. The effect of weather is represented by Fingrid's next-day
consumption, wind power and solar power forecasts. The Level 1 deliverable can be produced
entirely with the starter package data.
Starter package
Turku Energia provides the teams with a pre-loaded starter package (CSV files) containing
historical data from 1 June 2026 onwards at 15-minute resolution:
• imbalance price (the target variable to forecast) and mFRR and aFRR balancing data
• spot price (day-ahead), source Nord Pool
• Finnish consumption and production data, as well as consumption, wind power and solar power
forecasts
• cross-border flows between Finland and Sweden (SE1, SE3), Estonia and Norway, as well as the
available transmission capacity between Finland and Estonia
• imbalance prices and consumption forecasts of neighboring areas
• calendar variables (weekday, public holidays). In the package, the effect of weather is
represented by Fingrid's next-day consumption, wind power and solar power forecasts. The
package contains no separate weather data.
The package also includes a description of each file: source, unit, time zone and publication delay.
Every data row has a column first_usable_origin_utc, which states the first forecast origin at which
the row's information may be used. A forecast may only use information that would have been
available at the forecast origin.
Data outside the starter package
Teams may also use other open or paid data. Each team is itself responsible for the usage rights of
the data and lists all sources it has used in its submission. For example, processing of an ENTSO-E
Transparency Platform API token can take several working days, so it is worth applying for one
right away.

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 4

Primary open sources

Use in the challenge

In the starter
package

Source

Data

Fingrid

Imbalance price, mFRR and
aFRR balancing data,
consumption and production, Target variable to
consumption, wind power
forecast and the state Yes, CC BY
and solar power forecasts,
of the Finnish power 4.0
cross-border flows between system
Finland and Sweden (SE1,
SE3), Estonia and Norway

Imbalance prices and
consumption forecasts of
ENTSO-E
Market and system
neighboring areas (SE1, SE3,
Transparen
situation in
Estonia) and the transmission
cy Platform
neighboring areas
capacity between Finland
and Estonia

Yes, CC BY
4.0

Nord Pool

Spot price (day-ahead),
Finland

Yes. Data
licensed by
Turku
Energia: for
use during
the
Reference level for
hackathon
exceptional prices,
only, no
and the basis of the
redistributio
price in quarter-hours
n and no
with no dominant
publication
mFRR direction
in public
repositories.
The data will
be deleted
after the
event.

Finnish

Temperature, wind, solar

Effect of weather on

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

No. Teams

Page 5

Meteorologi
radiation
cal Institute
Calendar
data

consumption and on
wind and solar
production

can fetch
weather data
themselves.

Weekday, time of day, public Variation of
holidays, daylight saving
consumption and price Yes
time
over time

06 Evaluation Criteria
How should solutions be judged? What matters most — for example accuracy, scalability, usability, or
business impact?
Evaluation weighting: Level 1 = 80%, Level 2 = 20%. Success at Level 1 matters most; Level 2 brings
added value with which a team can stand out.
Level 1: Forecast quality and reasoning (80%)
➔ Forecast accuracy (30%): the forecast imbalance price compared with the actual price over the
test period.
➔ Uncertainty calibration (20%): how well the uncertainty interval (P10–P90) forecast by the team
describes the actual variation. With a well-calibrated forecast, about 80% of the actual prices fall
within the interval. Too narrow an interval means an overconfident forecast; too wide an interval
means a useless forecast. Scoring is based on the pinball loss of the P10 and P90 forecasts, as
described in the starter package.
➔ Identification of exceptional prices (20%): how well the model anticipates the quarter-hours in
which the imbalance price deviates significantly from the spot price (by more than 50 €/MWh
upwards or downwards).
➔ Direction of the balancing need (10%): accuracy of the forecast balancing direction (up / down /
no dominant direction). The direction means the dominant balancing direction of the quarter-hour
as defined by Fingrid, and it is based on mFRR activations. Source: Fingrid dataset 369
(dominant mFRR direction).
➔ Description of the reasoning chain (20%): clarity, consistency and inventiveness of the
description of how the data sources are used and of the factors affecting the forecast.
The percentages are shares of the Level 1 points. The first four sub-criteria are calculated with the scoring
script; the description of the reasoning chain is assessed by the jury. The jury assesses the description of
the reasoning chain on the basis of the evaluation report, the presentation slides and the demo video.
The forecast is not expected to be right all the time. What matters is that the uncertainty is calibrated and
that the solution recognizes situations in which the error may be large.
Forecast verification
Forecasts are evaluated over the test period. The starter package also contains the input data for the test

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 6

period. The actual values used in scoring are held only by the organizer. The test period will be
announced to the teams at the start of the event. A forecast may only use information that would have
been available at the forecast origin. A data row of the starter package may be used only from its
first_usable_origin_utc time onwards. If a row is used before this time, the submission is not a forecast,
and its automatic evaluation score is 0.
Scoring
➔ Two scripts are shared in the starter package: a validation script that checks the format of the
submission file, and a scoring script that calculates the metrics. The package also includes a
baseline forecast with its results.
➔ Each team checks its submission with the validation script before submitting. The validation does
not assess the quality of the forecast.
➔ Teams can calculate the metrics themselves for the validation period (7 Sep – 4 Oct 2026) with
the scoring script.
➔ The organizer calculates the official results against the actual values of the test period, with the
same scripts, for all submissions.
➔ The source code of the best solutions will be reviewed to ensure that no data has been used in
the forecasts before its usable time. AI may be used in the review.
Level 2: Situational overview and usability (20%)
➔ Clarity and user-friendliness of the user interface for an energy sector expert, for example a
production shift supervisor or an electricity trading specialist.
➔ Two time horizons: what is worth checking right now and over the next 10 hours, and how the
forecast has performed in hindsight (e.g. the previous day, week or month).
➔ The ability to move the point of review into the past, so that the usefulness of the solution can be
demonstrated over time and not only as real-time monitoring.
➔ Inventiveness: insights that give energy sector experts new information.

07 Submission Format
What deliverables do you expect? For example a demo video, pitch slides, a code repository, or
documentation.

The solution must produce a forecast of the imbalance price for the next 10 hours in 15-minute periods.
During the test period, the forecast is made on the hour. In addition to the price estimate, the forecast
must include an uncertainty interval, an estimate of the probability of significant price changes and the
most important factors affecting the forecast.
The hackathon does not require automatic trading, plant control or production-ready integrations. The
solution must be evaluable with the starter package data. Submissions are made via the Since AI
platform.
Level 1: The submission must include
➔ a machine-readable forecast file in the format described in the starter package, checked with the
validation script before submission
➔ source code and instructions for running it
➔ a short evaluation report: reasoning chain, factors affecting the forecast and the validation period

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 7

metrics
➔ a list of the data sources, external services, ready-made models and software components used,
and their usage rights
➔ about 5–8 presentation slides
➔ a demo video of at most 5 minutes, presenting the forecast for the next 10 hours in 15-minute
periods, the forecast uncertainty, significant deviations and the most important factors affecting
the forecast.
Level 2: Added value if the demo also shows
➔ a user-friendly view or alerts about situations in which the balancing need or the price forecast
changes materially
➔ how the forecast has performed in hindsight and the ability to move the point of review into the
past.
➔ the solution can produce a reasoned suggestion of situations in which Turku Energia should
reconsider its trading, production plan or use of capacity.

08 Point of Contact
Name, email and preferred contact method for questions during the hackathon.
Turku Energia coordination: Business Technology Manager Anne-Mari Angervo, annemari.angervo@turkuenergia.fi, 040 738 7073
Discord: te_ama_31257
Electricity market operations: Electricity Trading Specialist Ville Kuusisto, ville.kuusisto@turkuenergia.fi
Discord: te_ville_14208
Production: Production Manager Kalle Karjala, kalle.karjala@turkuenergia.fi
Discord: te_kalle
AI, data, technology and implementation: Development Specialist Vesa Lukka,
vesa.lukka@turkuenergia.fi
Discord: te_vesa_99724
In addition, from communications: Nora Forsman on Friday and Vilma Matero on Saturday/Sunday.
Fri 6 Nov from 15:30, EduCity
 on site: Ville, Kalle, Vesa and Ama + communications (Nora or Vilma)
 opening ceremony from 17:00, Since AI presents the challenges
 answering questions in our own challenge room (TE briefing room is 3006) from 18:50 until 20:00
 possibility of dinner
Sat 7 Nov
 Sat 9:00–12:00 // on site: Ville, Kalle, Ama
 Sat 14:30–18:00 // on site: Ville, Kalle, Vesa + communications
 possibility to eat during the event
 at other times we will answer as far as possible
Sun 8 Nov

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 8







8:45 arrival at EduCity, 9:00 briefing // on site: Ama, Vesa
Evaluation of solutions 10:00–13:00: places 1, 2 and 3 // on site: Ville, Kalle, Vesa, Ama
13:30 top 3 announced, final 14:00–15:30 // on site: everyone + children + communications
possibility to eat during the event

09 Glossary
Term

Plain-language explanation
Electricity must be produced at every moment in the same amount as it is
Power system balance consumed. If production and consumption do not match, the frequency of
the power system starts to deviate from its target level.
The difference between the planned and the actually realized electricity
Imbalance
production or consumption. For example, if a company has planned to
consume 100 units but consumes 110, an imbalance of 10 units arises.
A company or other actor that buys, sells, produces or consumes
Market party
electricity on the electricity market. Turku Energia acts in different roles
within this whole.
A market party that is financially responsible for ensuring that the
electricity procurement and delivery under its responsibility are in balance.
Balance responsible
Imbalances are settled in imbalance settlement. Turku Energia is not
party
necessarily the balance responsible party itself for every plant or
operation.
The price at which the imbalance between plan and actual is settled. The
Imbalance price
price may be unfavorable compared with having bought or sold the
required electricity on the market in time.
Production or consumption capacity reserved in advance for use, which
can be activated quickly to balance the power system. For example, a
Reserve
power plant may commit to being able to increase or decrease its output
when needed.
Electricity that is added or reduced to balance the system. Upward
Balancing energy
regulation means in practice increasing production or reducing
consumption, and downward regulation the opposite. Same as mFRR.
Different reserves. aFRR is the automatically activated frequency
restoration reserve and mFRR the manually activated frequency
mFRR and aFRR
restoration reserve. They respond in different ways and are handled on
different markets.
The bidding zone price usually formed on the next-day electricity market,
Spot price
i.e. the day-ahead market. It should not be confused with intraday prices
or imbalance prices.
A market on which electricity is bought and sold for the hours or quarterDay-ahead market
hour periods of the next day, based on the forecasts available.
The within-day market on which earlier plans can be corrected closer to
Intraday market
the actual delivery time of the electricity.
Asset, i.e. a production A physical resource whose production or consumption can be controlled.
or consumption
A production asset can be, for example, a power plant; a consumption
resource
asset, for example, an electric boiler, a heat pump, a battery or a flexible

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 9

Term
Imbalance cost
Demand response

Plain-language explanation
industrial load.
The financial cost arising from an imbalance. It depends on the volume,
the direction and the price of the imbalance.
Increasing, reducing or shifting consumption to another time based on the
price or the needs of the power system. Total consumption does not
necessarily have to decrease; it can instead be timed better.

Since AI ry · Business ID 3593920-2 · Vähä Hämeenkatu 1 A 3, 20500 Turku, Finland

Page 10

