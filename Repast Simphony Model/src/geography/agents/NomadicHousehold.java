package geography.agents;

import java.util.*;
import java.util.stream.Collectors;
import org.locationtech.jts.geom.*;
import repast.simphony.engine.environment.RunEnvironment;
import repast.simphony.engine.schedule.ScheduledMethod;
import repast.simphony.parameter.Parameters;
import repast.simphony.random.RandomHelper;
import repast.simphony.space.gis.Geography;

public class NomadicHousehold {

    private String id;
    private Geography<Object> geography;
    private ModelManager modelManager;
    private GeometryFactory geomFactory = new GeometryFactory();

    private double currentRainfall;
    private double currentSavi;
    private double previousTickX = -1;
    private double previousTickY = -1;

    private double distanceMovedKm = 0.0;
    private String movementPathWKT = "";

    private String movementStrategy;
    private double maxSearchRadiusKm;
    
    private double rainThreshold; 
    private double saviThreshold; 
    
    private double visitedWeight;
    private double randomWeight;
    private double pMove;

    private final double MIN_SEARCH_RADIUS_KM = 15.0;
    private final double SEARCH_STEP_METERS = 2000.0; 
    
    private static int lastLoggedMonth = -1;

    public NomadicHousehold(String id, Geography<Object> geography, ModelManager modelManager) {
        this.id = id;
        this.geography = geography;
        this.modelManager = modelManager;
        
        Parameters params = RunEnvironment.getInstance().getParameters();
        this.movementStrategy = params.getString("movementStrategy");
        
        if (this.movementStrategy.contains("60km")) this.maxSearchRadiusKm = 60.0;
        else this.maxSearchRadiusKm = 30.0;

        this.rainThreshold = params.getDouble("rainThreshold");
        this.saviThreshold = params.getDouble("saviThreshold");
        
        this.visitedWeight = params.getDouble("visitedWeight");
        this.randomWeight = params.getDouble("randomWeight");
        this.pMove = params.getDouble("pMove");
    }

    public void initializeLocation() {
        Point p = (Point) geography.getGeometry(this);
        if (p != null) {
            this.previousTickX = p.getX();
            this.previousTickY = p.getY();
            // Register in spatial hash (5m rule)
            modelManager.addOccupiedCoordinate(p.getCoordinate());
            updateCurrentEnvironmentalValues(0);
        }
    }

    @ScheduledMethod(start = 1, interval = 1)
    public void step() {
        this.distanceMovedKm = 0.0;
        this.movementPathWKT = "";

        try {
            double tick = RunEnvironment.getInstance().getCurrentSchedule().getTickCount();
            int monthIndex = (int) tick - 1;
            if (monthIndex < 0) monthIndex = 0;
            if (monthIndex > 11) monthIndex = 11; 

            if (monthIndex != lastLoggedMonth) {
                System.out.println("==================================================");
                System.out.println(">>> COMPUTING TICK " + (monthIndex + 1) + " (Month Index: " + monthIndex + ") <<<");
                System.out.println("==================================================");
                lastLoggedMonth = monthIndex;
            }

            Point currentPoint = (Point) geography.getGeometry(this);
            if (currentPoint == null) return;

            updateCurrentEnvironmentalValues(monthIndex);

            if (this.movementStrategy.startsWith("RainfallPriority")) {
                executeRainfallPriority(currentPoint, monthIndex);
            } else {
                executePasturePriority(currentPoint, monthIndex);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void executeRainfallPriority(Point currentPoint, int month) {
        List<CandidateCell> candidates = getCandidateCells(currentPoint.getCoordinate(), month);
        if (candidates.isEmpty()) return; 

        List<CandidateCell> betterRain = candidates.stream()
                .filter(c -> c.rainfall > (this.currentRainfall + this.rainThreshold))
                .collect(Collectors.toList());

        if (betterRain.isEmpty()) return; 

        double maxRain = betterRain.stream().mapToDouble(c -> c.rainfall).max().orElse(-1.0);
        List<CandidateCell> bestRainCells = betterRain.stream()
                .filter(c -> Math.abs(c.rainfall - maxRain) < 0.001)
                .collect(Collectors.toList());

        if (bestRainCells.size() == 1) {
            attemptMove(bestRainCells.get(0).coord, month);
            return;
        }

        List<CandidateCell> betterSavi = bestRainCells.stream()
                .filter(c -> c.savi > (this.currentSavi + this.saviThreshold))
                .collect(Collectors.toList());
        
        List<CandidateCell> nextStepCandidates;
        
        if (betterSavi.isEmpty()) {
            nextStepCandidates = bestRainCells; 
        } else {
            double bestSaviVal = betterSavi.stream().mapToDouble(c -> c.savi).max().orElse(-1.0);
            nextStepCandidates = betterSavi.stream()
                    .filter(c -> Math.abs(c.savi - bestSaviVal) < 0.0001)
                    .collect(Collectors.toList());
        }

        if (nextStepCandidates.size() == 1) {
            attemptMove(nextStepCandidates.get(0).coord, month);
            return;
        }

        double maxWater = nextStepCandidates.stream().mapToDouble(c -> c.water).max().orElse(0.0);
        List<CandidateCell> finalCandidates;
        
        if (maxWater == 0) {
            // V7 Rule: If W_proximity_max = 0, skip this step entirely
            finalCandidates = nextStepCandidates; 
        } else {
            finalCandidates = nextStepCandidates.stream()
                    .filter(c -> Math.abs(c.water - maxWater) < 0.0001)
                    .collect(Collectors.toList());
        }
        
        if (finalCandidates.size() == 1) {
            attemptMove(finalCandidates.get(0).coord, month);
            return;
        }

        CandidateCell winner = resolveTies(finalCandidates, currentPoint.getCoordinate());
        if (winner != null) attemptMove(winner.coord, month);
    }

    private void executePasturePriority(Point currentPoint, int month) {
        List<CandidateCell> candidates = getCandidateCells(currentPoint.getCoordinate(), month);
        if (candidates.isEmpty()) return;

        List<CandidateCell> betterSavi = candidates.stream()
                .filter(c -> c.savi > (this.currentSavi + this.saviThreshold))
                .collect(Collectors.toList());

        List<CandidateCell> nextStepCandidates;
        boolean skipWaterCheck = false;

        // V6 FIX: If SAVI doesn't improve, cascade to Rainfall Fallback
        if (betterSavi.isEmpty()) {
            List<CandidateCell> betterRain = candidates.stream()
                    .filter(c -> c.rainfall > (this.currentRainfall + this.rainThreshold))
                    .collect(Collectors.toList());
            
            if (betterRain.isEmpty()) return; // Agent stays

            double maxRain = betterRain.stream().mapToDouble(c -> c.rainfall).max().orElse(-1.0);
            nextStepCandidates = betterRain.stream()
                    .filter(c -> Math.abs(c.rainfall - maxRain) < 0.001)
                    .collect(Collectors.toList());
        } else {
            double maxSavi = betterSavi.stream().mapToDouble(c -> c.savi).max().orElse(-1.0);
            List<CandidateCell> bestSaviCells = betterSavi.stream()
                    .filter(c -> Math.abs(c.savi - maxSavi) < 0.0001)
                    .collect(Collectors.toList());
                    
            // 🚨 V7 NEW: Step 2.5 Rainfall Check on SAVI-Improved Cells
            List<CandidateCell> saviAndRain = bestSaviCells.stream()
                    .filter(c -> c.rainfall > (this.currentRainfall + this.rainThreshold))
                    .collect(Collectors.toList());
                    
            if (!saviAndRain.isEmpty()) {
                nextStepCandidates = saviAndRain;
                skipWaterCheck = true; // V7 Rule: Proceed to Step 5 (Skip water check)
            } else {
                nextStepCandidates = bestSaviCells; // Keep all cells from Step 2
            }
        }

        if (nextStepCandidates.size() == 1 && skipWaterCheck) {
            attemptMove(nextStepCandidates.get(0).coord, month);
            return;
        }

        List<CandidateCell> finalCandidates;
        
        if (!skipWaterCheck) {
            double maxWater = nextStepCandidates.stream().mapToDouble(c -> c.water).max().orElse(0.0);
            
            if (maxWater == 0) {
                // 🚨 V7 HARD BLOCK: Agent STAYS (Movement blocked to prevent risk)
                return; 
            } else {
                finalCandidates = nextStepCandidates.stream()
                        .filter(c -> Math.abs(c.water - maxWater) < 0.0001)
                        .collect(Collectors.toList());
            }
        } else {
            finalCandidates = nextStepCandidates;
        }

        CandidateCell winner = resolveTies(finalCandidates, currentPoint.getCoordinate());
        if (winner != null) attemptMove(winner.coord, month);
    }

    private CandidateCell resolveTies(List<CandidateCell> cells, Coordinate currentCoord) {
        if (cells.isEmpty()) return null;
        CandidateCell bestCell = null;
        double bestScore = -Double.MAX_VALUE;

        for (CandidateCell c : cells) {
            double distKm = currentCoord.distance(c.coord) / 1000.0;
            double distNorm = distKm / this.maxSearchRadiusKm;
            boolean visited = modelManager.isVisited(c.coord, this.id);
            double visitedVal = visited ? 1.0 : 0.0;
            double randVal = RandomHelper.nextDouble(); 

            double score = -distNorm + (visitedWeight * visitedVal) + (randomWeight * randVal);
            if (score > bestScore) {
                bestScore = score;
                bestCell = c;
            }
        }
        return bestCell;
    }

    private void attemptMove(Coordinate targetGrid, int month) {
        if (RandomHelper.nextDouble() > pMove) return; 

        double destX = 0;
        double destY = 0;
        boolean foundSpot = false;
        
        for (int i = 0; i < 15; i++) { 
            double angle = RandomHelper.nextDouble() * 2 * Math.PI;
            double radius = RandomHelper.nextDouble() * (SEARCH_STEP_METERS * 0.75); 
            
            destX = targetGrid.x + (radius * Math.cos(angle));
            destY = targetGrid.y + (radius * Math.sin(angle));
            
            if (!modelManager.isValidLocation(destX, destY)) continue;
            if (modelManager.isTooCloseToOthers(destX, destY, 5.0)) continue;
            
            foundSpot = true;
            break;
        }
        
        if (!foundSpot) return; 

        Point oldPoint = (Point) geography.getGeometry(this);
        Point newPoint = geomFactory.createPoint(new Coordinate(destX, destY));
        
        this.distanceMovedKm = oldPoint.getCoordinate().distance(newPoint.getCoordinate()) / 1000.0;
        this.movementPathWKT = String.format("LINESTRING (%.6f %.6f, %.6f %.6f)",
                oldPoint.getX(), oldPoint.getY(), destX, destY);

        geography.move(this, newPoint);
        
        if (this.id.equals("Household_0")) {
            System.out.printf("   >>> Household_0 successfully moved to [%.0f, %.0f]\n", destX, destY);
        }
        
        modelManager.removeOccupiedCoordinate(oldPoint.getCoordinate());
        modelManager.addOccupiedCoordinate(newPoint.getCoordinate());
        
        modelManager.markVisited(newPoint, this.id);
        modelManager.trackExit(this, month + 1); 
        
        this.previousTickX = destX;
        this.previousTickY = destY;
    }

    private List<CandidateCell> getCandidateCells(Coordinate current, int month) {
        List<CandidateCell> list = new ArrayList<>();
        double minR_Meters = MIN_SEARCH_RADIUS_KM * 1000.0;
        double maxR_Meters = maxSearchRadiusKm * 1000.0;
        
        double startX = Math.floor((current.x - maxR_Meters) / SEARCH_STEP_METERS) * SEARCH_STEP_METERS;
        double endX = Math.ceil((current.x + maxR_Meters) / SEARCH_STEP_METERS) * SEARCH_STEP_METERS;
        double startY = Math.floor((current.y - maxR_Meters) / SEARCH_STEP_METERS) * SEARCH_STEP_METERS;
        double endY = Math.ceil((current.y + maxR_Meters) / SEARCH_STEP_METERS) * SEARCH_STEP_METERS;

        ModelManager.DirectRaster rainR = modelManager.getRainfallRaster(month);
        ModelManager.DirectRaster saviR = modelManager.getSaviRaster(month);
        ModelManager.DirectRaster rangelandR = modelManager.getRangelandsRaster();
        ModelManager.DirectRaster slopeR = modelManager.getSlopeRaster();

        for (double x = startX; x <= endX; x += SEARCH_STEP_METERS) {
            for (double y = startY; y <= endY; y += SEARCH_STEP_METERS) {
                
                double dx = x - current.x;
                double dy = y - current.y;
                double distSq = dx*dx + dy*dy;
                
                if (distSq >= (minR_Meters*minR_Meters) && distSq <= (maxR_Meters*maxR_Meters)) {
                    
                    double r = rainR.getValue(x, y);
                    if (r == ModelManager.NODATA_VALUE) continue; 
                    
                    if (Math.abs(rangelandR.getValue(x, y) - 1.0) > 0.001) continue;
                    if (Math.abs(slopeR.getValue(x, y) - 1.0) > 0.001) continue;
                    
                    if (!modelManager.isNonSedentary(x, y)) continue;

                    double s = saviR.getValue(x, y);
                    
                    // 🚨 V7 FIX: W_proximity_i (Max water within 10km buffer)
                    double w = modelManager.getMaxWaterInRadius(x, y, 10000.0);
                    
                    list.add(new CandidateCell(new Coordinate(x, y), r, s, w));
                }
            }
        }
        return list;
    }

    private void updateCurrentEnvironmentalValues(int month) {
        Point p = (Point) geography.getGeometry(this);
        if (p == null) return;
        this.currentRainfall = modelManager.getRasterValueAtCoord(p.getX(), p.getY(), modelManager.getRainfallRaster(month));
        this.currentSavi = modelManager.getRasterValueAtCoord(p.getX(), p.getY(), modelManager.getSaviRaster(month));
    }
    
    private class CandidateCell {
        Coordinate coord;
        double rainfall, savi, water;
        CandidateCell(Coordinate c, double r, double s, double w) {
            this.coord = c; this.rainfall = r; this.savi = s; this.water = w;
        }
    }
    
    public String getId() { return id; }
    public double getTick() { return RunEnvironment.getInstance().getCurrentSchedule().getTickCount(); }
    public double getX() { return previousTickX; }
    public double getY() { return previousTickY; }
    public boolean isMoving() { return distanceMovedKm > 0; }
    public double getDistanceMovedKm() { return distanceMovedKm; }
    public double getCurrentRainfall() { return currentRainfall; }
    public double getCurrentSavi() { return currentSavi; }
    public String getMovementPathWKT() { return movementPathWKT; }
    public String getMovementStrategy() { return movementStrategy; }
    public int getExitMonth() { Integer m = modelManager.getAgentExitMonth(id); return (m != null) ? m : -1; }
}